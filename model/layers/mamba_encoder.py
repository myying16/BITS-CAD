from torch import nn
from model.mamba_ssm.modules.mamba_simple import Mamba
from model.model_utils import _make_batch_first, _make_seq_first


class EncoderMambaBiBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mamba = Mamba1(
            d_model=cfg.d_model,
            #text_dim=cfg.d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            bimamba=True,
            use_fast_path=False,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4*cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4*cfg.d_model, cfg.d_model),
        )
    def forward(self, x_snd,text, text_mask=None,padding_mask=None):
        # x_snd: (S,B,D),text: 1,N,D-B,L,D
        if padding_mask is not None:
            x_snd = x_snd * padding_mask
        h = self.norm(x_snd)

        h_bld = _make_batch_first(h)        
        
        y_bld = self.mamba(h_bld,text,text_mask=text_mask)           
        y_snd = _make_seq_first(y_bld)        

        x_snd = x_snd + self.drop(y_snd)
        if padding_mask is not None:
            x_snd = x_snd * padding_mask
        h2 = self.ffn_norm(x_snd)
        x_snd = x_snd + self.ffn(h2)
        if padding_mask is not None:
            x_snd = x_snd * padding_mask
        return x_snd
