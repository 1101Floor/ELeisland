import torch
from torch import nn
from einops import rearrange

from layers.gumbel_sinkhorn import GumbelSinkhornOrdering
from layers.Embed import  OrderAwarePatchEmbedding
from layers.Mamba_Encoder import MambaEncoder


class PredictionHead(nn.Module):
    """预测头: Conv1d 版本"""

    def __init__(self, d_model, pred_len, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=pred_len,
            kernel_size=num_patches,
            stride=num_patches,
            bias=True,
        )

    def forward(self, x, n_vars):
        # x: [B, n_vars * num_patches, d_model]
        x = rearrange(x, 'b (c p) d -> (b c) d p', c=n_vars, p=self.num_patches)
        x = self.conv(x)  # [B*n_vars, pred_len, 1]
        x = rearrange(x, '(b c) p 1 -> b p c', c=n_vars)
        return x


class Model(nn.Module):


    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.n_vars = configs.enc_in

        num_patches = (configs.seq_len - configs.patch_len) // configs.stride + 1
        self.num_patches = num_patches

        # ---- Gumbel-Sinkhorn ----
        self.ordering_net = GumbelSinkhornOrdering(
            d_input=configs.seq_len,
            n_vars=configs.enc_in,
            d_hidden=getattr(configs, 'gs_d_hidden', 32),
            n_sinkhorn_iters=getattr(configs, 'gs_sinkhorn_iters', 10),
            temperature_init=getattr(configs, 'gs_temperature', 1.0),
            dropout=configs.dropout,
        )


        self.embedding = OrderAwarePatchEmbedding(
            patch_len=configs.patch_len,
            d_model=configs.d_model,
            n_vars=configs.enc_in,
            num_patches=num_patches,
            order_bias_weight=getattr(configs, 'order_bias_weight', 1.0),
        )

        # ---- Encoder (纯 Mamba) ----
        self.encoder =MambaEncoder(
            d_model=configs.d_model,
            n_layers=configs.e_layers,
            dropout=configs.dropout,
        )

        # ---- Prediction Head ----
        self.head = PredictionHead(
            configs.d_model, configs.pred_len, num_patches
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,return_perm=False):
        b, seq_len, n_vars = x_enc.shape

        # ---- Step 1: 归一化 ----

        
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.std(
            x_enc, dim=1, keepdim=True, unbiased=False
        ).clamp(min=1e-5).detach()
        x_enc = x_enc / stdev
        

        # ---- Step 2: Gumbel-Sinkhorn ----
        x_for_order = x_enc.permute(0, 2, 1)  # [B, n_vars, seq_len]
        x_ordered, perm_matrix = self.ordering_net(
            x_for_order,
            hard= not self.training,
            return_extras=True,
        )

        # ---- Step 3: Patch 切分 ----
        x_patches = x_ordered.unfold(-1, self.patch_len, self.stride)
        # x_patches: [B, n_vars, num_patches, patch_len]

        # ---- Step 4: Embedding (patch投影  + order_bias + VST) ----
        enc_in = self.embedding(x_patches, perm_matrix)
        # enc_in: [B, n_vars * num_patches, d_model]

        # ---- Step 5: Mamba Encoder (纯扫描) ----
        enc_out = self.encoder(enc_in)
        # enc_out: [B, n_vars * num_patches, d_model]

        # ---- Step 6: Prediction Head ----
        dec_out = self.head(enc_out, n_vars=n_vars)
        # dec_out: [B, pred_len, n_vars]

        # ---- Step 7: 反归一化 ----

        
        dec_out = dec_out * stdev.expand(-1, self.pred_len, -1)
        dec_out = dec_out + means.expand(-1, self.pred_len, -1)

        if return_perm:
            return dec_out, perm_matrix
        else:
            return dec_out
