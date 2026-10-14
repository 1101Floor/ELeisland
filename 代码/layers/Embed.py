import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class OrderAwarePatchEmbedding(nn.Module):


    def __init__(
        self,
        patch_len: int,
        d_model: int,
        n_vars: int,
        num_patches: int,
        order_bias_weight: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.num_patches = num_patches
        self.order_bias_weight = order_bias_weight

        # ---- Patch → d_model 映射 ----
        self.patch_proj = nn.Linear(patch_len, d_model, bias=False)


        # ----顺序感知偏置 ----
        # 每个"邻居位置 k"对应一个 d_model 维可学习向量
        # 不同邻居身份产生不同方向的偏置
        self.order_bias = nn.Parameter(
            torch.randn(n_vars, d_model) * 0.01
        )

    def _compute_adjacency(self, perm_matrix: torch.Tensor) -> torch.Tensor:
        """
        基于软置换矩阵计算因果前驱邻接矩阵（仅捕捉每个变量的前一位置邻居，保证无自环）。

        Args:
            perm_matrix P: Tensor with shape (B, n_vars, n_vars)
                P[b, i, j]：第b个样本中，变量i被分配到序列位置j的概率

        Returns:
            adj: Tensor with shape (B, n_vars, n_vars)
                adj[b, i, k] 代表联合概率：变量k排在变量i的前一个位置；
                对角线恒为0，消除变量自环（变量不能是自身前驱）
        """
        B, n_vars, _ = perm_matrix.shape

        # P_bwd[:, k, j] = perm_matrix[:, k, j-1]；j=0位置填充0，无前置变量 前向感知

        P_bwd = F.pad(perm_matrix[:, :, :-1], pad=(1, 0))

        #  后向感知
        # P_bwd = F.pad(perm_matrix[:, :, 1:], pad=(0, 1))


        # 批量矩阵乘法：adj[b,i,k] = sum_{j} P[b,i,j] * P_bwd[b,k,j]
        adj = torch.bmm(perm_matrix, P_bwd.transpose(1, 2))


        return adj

    def forward(
        self,
        x_patches: torch.Tensor,
        perm_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_patches:  [B, n_vars, num_patches, patch_len]
            perm_matrix:[B, n_vars, n_vars]

        Returns:
            [B, n_vars * num_patches, d_model]
        """
        B, C, P, L = x_patches.shape
        assert C == self.n_vars and P == self.num_patches


        # [B, n_vars, num_patches, patch_len] → [B, n_vars, num_patches, d_model]
        x = self.patch_proj(x_patches)


        # ---- 顺序感知偏置 ----
        adj = self._compute_adjacency(perm_matrix)
        # adj: [B, n_vars, n_vars]

        # 每个变量根据其前一个邻居的身份获得偏置
        order_bias = torch.bmm(
            adj,
            self.order_bias.unsqueeze(0).expand(B, -1, -1),
        )
        # order_bias: [B, n_vars, d_model]

        x = x + order_bias.unsqueeze(2) * self.order_bias_weight
        # shape 不变: [B, n_vars, num_patches, d_model]

        # ---- VST 重排 ----
        # 将变量和 patch 交错排列为一维序列
        x = rearrange(x, 'b c p d -> b (c p) d')
        # x: [B, n_vars * num_patches, d_model]

        return x
