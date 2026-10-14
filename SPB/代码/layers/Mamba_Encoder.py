import torch
import torch.nn as nn
from mamba_ssm import Mamba



class MambaEncoder(nn.Module):


    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model



        # Mamba 层堆叠
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, L, D = x.shape

        # ---- Mamba 层 ----
        for layer in self.layers:
            x = x + layer(x)

        return self.final_norm(x)
