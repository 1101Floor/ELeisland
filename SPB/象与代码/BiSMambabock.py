# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified to support bidirectional processing
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BiSMambaBlock(nn.Module):
    """
    双向Mamba块，通过并行处理正向和反向序列来捕获双向上下文信息（drop版本）
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
            dropout=0.1,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.dropout = nn.Dropout(dropout)

        # 输入输出投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.activation = "silu"
        self.act = nn.SiLU()

        # 正向和反向的状态投影
        self.forward_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.backward_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        # 正向和反向的时间步长投影
        self.forward_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.backward_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # 初始化时间步长投影
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.forward_dt_proj.weight, dt_init_std)
            nn.init.constant_(self.backward_dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.forward_dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.backward_dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.forward_dt_proj.bias.copy_(inv_dt)
            self.backward_dt_proj.bias.copy_(inv_dt)

        self.forward_dt_proj.bias._no_reinit = True
        self.backward_dt_proj.bias._no_reinit = True

        # S4D真实初始化 - 正向
        forward_A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(self.d_inner, 1).contiguous()
        forward_A_log = torch.log(forward_A)
        self.forward_A_log = nn.Parameter(forward_A_log)
        self.forward_A_log._no_weight_decay = True

        # S4D真实初始化 - 反向
        backward_A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(self.d_inner, 1).contiguous()
        backward_A_log = torch.log(backward_A)
        self.backward_A_log = nn.Parameter(backward_A_log)
        self.backward_A_log._no_weight_decay = True

        # D "skip"参数
        self.forward_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.backward_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.forward_D._no_weight_decay = True
        self.backward_D._no_weight_decay = True


    def _forward_mamba(self, x, z, x_proj, dt_proj, A_log, D, is_backward=False):
        """
        单向Mamba处理函数
        输入:
          x: (B, d_inner, L)
          z: (B, d_inner, L)
        返回:
          y: (B, d_inner, L)
        """
        batch, d_inner, seqlen = x.shape

        # 投影得到状态参数
        x_dbl = x_proj(x.transpose(1, 2).reshape(batch * seqlen, d_inner))  # (B*l, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 计算 dt
        dt = dt_proj.weight @ dt.t()  # (d_inner, B*l)
        dt = dt.reshape(d_inner, batch, seqlen).permute(1, 0, 2)  # (B, d_inner, L)

        # 恢复 B,C 的形状
        B = B.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        C = C.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()

        # A 为负的指数
        A = -torch.exp(A_log.float())  # (d_inner, d_state)

        # 选择性扫描
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )

        return y

    def forward(self, hidden_states: Tensor, inference_params=None) -> Tensor:
        """
        双向Mamba前向传播
        """
        batch, seqlen, dim = hidden_states.shape

        # 推理分支
        if inference_params is not None:
            forward_ssm_state, backward_ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, forward_ssm_state, backward_ssm_state = self.step(
                    hidden_states, forward_ssm_state, backward_ssm_state)
                return out

        # 输入投影
        xz = self.in_proj(hidden_states).transpose(1, 2)

        x, z = xz.chunk(2, dim=1)  # (B, d_inner, L)

        # 应用时间特征dropout
        x = self.dropout(x)

        # 准备反向分支：在序列维度翻转

        xz_rev = self.in_proj(hidden_states).transpose(1, 2)
        xz_rev = torch.flip(xz_rev, dims=[1,2])  # 在L，D维度翻转
        x_rev, z_rev = xz_rev.chunk(2, dim=1)

        x_rev = self.dropout(x_rev)
        # 正向处理
        forward_y = self._forward_mamba(
            x, z,
            self.forward_x_proj,
            self.forward_dt_proj,
            self.forward_A_log,
            self.forward_D,
            is_backward=False
        )

        # 反向处理
        backward_y_rev = self._forward_mamba(
            x_rev, z_rev,
            self.backward_x_proj,
            self.backward_dt_proj,
            self.backward_A_log,
            self.backward_D,
            is_backward=True
        )

        # 将反向结果翻转回原始顺序
        backward_y = torch.flip(backward_y_rev, dims=[1,2])  # 在L D 维度翻转


        # 融合正向和反向结果
        combined_y = forward_y + backward_y
        combined_y = combined_y * self.act(z)# 使用z进行门控

        # 输出投影
        combined_y = combined_y.transpose(1, 2)
        out = self.out_proj(combined_y)

        return out


    def step(self, hidden_states, forward_ssm_state, backward_ssm_state):
        """
        单步推理
        输入:
          hidden_states: (B, 1, D)
          forward_ssm_state: (B, d_inner, d_state)
          backward_ssm_state: (B, d_inner, d_state)
        返回:
          out: (B, 1, D) 以及更新后的状态
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1

        # 输入投影
        xz = self.in_proj(hidden_states.squeeze(1))  # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 各为 (B, d_inner)

        # 正向单步
        forward_x = x
        forward_z = z
        forward_x_db = self.forward_x_proj(forward_x)
        forward_dt, forward_B, forward_C = torch.split(
            forward_x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        forward_dt = F.linear(forward_dt, self.forward_dt_proj.weight)
        forward_A = -torch.exp(self.forward_A_log.float())

        if selective_state_update is None:
            forward_dt = F.softplus(forward_dt + self.forward_dt_proj.bias.to(dtype=forward_dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", forward_dt, forward_A))
            dB = torch.einsum("bd,bn->bdn", forward_dt, forward_B)
            forward_ssm_state.copy_(forward_ssm_state * dA + forward_x.unsqueeze(-1) * dB)
            forward_y = torch.einsum("bdn,bn->bd", forward_ssm_state.to(dtype), forward_C)
            forward_y = forward_y + self.forward_D.to(dtype) * forward_x
        else:
            forward_y = selective_state_update(
                forward_ssm_state, forward_x, forward_dt,
                forward_A, forward_B, forward_C, self.forward_D,
                z=forward_z, dt_bias=self.forward_dt_proj.bias, dt_softplus=True
            )

        # 反向单步（使用相同的输入，因为反向序列已在forward中处理）
        backward_x = x
        backward_z = z
        backward_x_db = self.backward_x_proj(backward_x)
        backward_dt, backward_B, backward_C = torch.split(
            backward_x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        backward_dt = F.linear(backward_dt, self.backward_dt_proj.weight)
        backward_A = -torch.exp(self.backward_A_log.float())

        if selective_state_update is None:
            backward_dt = F.softplus(backward_dt + self.backward_dt_proj.bias.to(dtype=backward_dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", backward_dt, backward_A))
            dB = torch.einsum("bd,bn->bdn", backward_dt, backward_B)
            backward_ssm_state.copy_(backward_ssm_state * dA + backward_x.unsqueeze(-1) * dB)
            backward_y = torch.einsum("bdn,bn->bd", backward_ssm_state.to(dtype), backward_C)
            backward_y = backward_y + self.backward_D.to(dtype) * backward_x
        else:
            backward_y = selective_state_update(
                backward_ssm_state, backward_x, backward_dt,
                backward_A, backward_B, backward_C, self.backward_D,
                z=backward_z, dt_bias=self.backward_dt_proj.bias, dt_softplus=True
            )

        # 融合正向和反向结果
        combined_y = forward_y + backward_y
        combined_y = combined_y * self.act(z)
        out = self.out_proj(combined_y)

        return out.unsqueeze(1), forward_ssm_state, backward_ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        为推理分配状态缓存
        """
        device = self.out_proj.weight.device
        if dtype is None:
            dtype = self.forward_dt_proj.weight.dtype

        # 正向状态
        forward_ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=device,
            dtype=dtype
        )

        # 反向状态
        backward_ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=device,
            dtype=dtype
        )

        return forward_ssm_state, backward_ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """从缓存中获取状态"""
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            # 初始化状态
            forward_ssm_state, backward_ssm_state = self.allocate_inference_cache(batch_size, 0)
            inference_params.key_value_memory_dict[self.layer_idx] = (
                forward_ssm_state, backward_ssm_state
            )
        else:
            # 从缓存获取状态
            forward_ssm_state, backward_ssm_state = \
                inference_params.key_value_memory_dict[self.layer_idx]

        # 初始化状态（如果需要）
        if initialize_states:
            forward_ssm_state.zero_()
            backward_ssm_state.zero_()

        return forward_ssm_state, backward_ssm_state