from torch import nn
from model.mamba_ssm.modules.mamba_simple import Mamba
from model.model_utils import _make_batch_first, _make_seq_first


class EncoderMambaBiBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mamba = Mamba(
            d_model=cfg.d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            bimamba=True,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4*cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4*cfg.d_model, cfg.d_model),
        )
    def forward(self, x_snd, padding_mask=None):
        # x_snd: (S,N,D)
        if padding_mask is not None:
            x_snd = x_snd * padding_mask
        h = self.norm(x_snd)

        h_bld = _make_batch_first(h)               # (N,S,D)
        y_bld = self.mamba(h_bld)           # (N,S,D) 由 mamba_simple.forward 决定
        y_snd = _make_seq_first(y_bld)           # (S,N,D)

        x_snd = x_snd + self.drop(y_snd)
        if padding_mask is not None:
            x_snd = x_snd * padding_mask
        h2 = self.ffn_norm(x_snd)
        x_snd = x_snd + self.ffn(h2)
        if padding_mask is not None:
            x_snd = x_snd * padding_mask
        return x_snd