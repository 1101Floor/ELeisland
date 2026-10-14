
import math
import torch
import torch.nn as nn
from einops import rearrange

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20,
             temperature: float = 1.0) -> torch.Tensor:
    """Sinkhorn 迭代: 实数矩阵 → 双随机矩阵"""
    log_alpha = log_alpha / temperature
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def to_hard_perm(soft_perm: torch.Tensor) -> torch.Tensor:

    if not HAS_SCIPY:
        raise ImportError("Scipy")
    batch_size, n, _ = soft_perm.shape
    hard_perm = torch.zeros_like(soft_perm)
    for b in range(batch_size):
        cost = -soft_perm[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        hard_perm[b, row_ind, col_ind] = 1.0
    return hard_perm.to(soft_perm.device)


class GumbelSinkhornOrdering(nn.Module):
    """排序网络: 学习变量的最优排列顺序"""

    def __init__(
        self,
        d_input: int,
        n_vars: int,
        d_hidden: int = 128,
        n_sinkhorn_iters: int = 20,
        temperature_init: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.d_hidden = d_hidden

        self.variable_encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.position_embeddings = nn.Parameter(
            torch.randn(n_vars, d_hidden) * (1.0 / math.sqrt(d_hidden))
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.5 , max=5.0)

    def forward(self, x: torch.Tensor, hard: bool = False,
                return_extras: bool = False):
        """
        Args:
            x: (batch, n_vars, d_input) — 每个变量的原始特征
            hard: True 时用匈牙利算法输出硬排列（推理用）
            return_extras: 是否返回中间信息（perm_matrix, log_alpha 等）

        Returns:
            x_reordered: (batch, n_vars, d_input) — 重排后的变量
            perm_matrix: (batch, n_vars, n_vars) — 排列矩阵
            extras: dict（可选）
        """
        # 1. 编码每个变量的特征
        var_features = self.variable_encoder(x)  # [B, n_vars, d_hidden]

        # 2. 变量特征 × 位置嵌入 → 得分矩阵
        log_alpha = torch.matmul(var_features, self.position_embeddings.T)
        # [B, n_vars, d_hidden] × [d_hidden, n_vars] → [B, n_vars, n_vars]


        # 3. 训练时加 Gumbel 噪声（探索不同排列）
        if self.training:
            u = torch.rand_like(log_alpha).clamp(1e-8, 1.0 - 1e-8)
            log_alpha = log_alpha + (-torch.log(-torch.log(u)))

        # 4. Sinkhorn → 软排列矩阵
        temperature = self.get_temperature()
        perm_matrix = sinkhorn(log_alpha, self.n_sinkhorn_iters, temperature)

        # 5. 推理时转硬排列
        if hard:
            perm_matrix = to_hard_perm(perm_matrix)

        # 6. 用排列矩阵重排输入
        x_reordered = torch.bmm(perm_matrix, x)


        return x_reordered, perm_matrix
