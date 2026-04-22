from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from .attention import MultiheadAttention
from .transformer import _get_activation_fn


#TransformerEncoderLayerImproved，TransformerDecoderLayerImproved，TransformerDecoderLayerGlobalImproved


#SVG编码器：增强的编码器层，支持额外全局输入memory2
class TransformerEncoderLayerImproved(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", d_global2=None):
        super(TransformerEncoderLayerImproved, self).__init__()
        # 1. 自注意力层
        self.self_attn = MultiheadAttention(d_model,d_model, nhead, dropout=dropout)
        # 2. 可选的第二全局输入处理
        if d_global2 is not None:
            self.linear_global2 = Linear(d_global2, d_model)
        # 3. 前馈网络Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        # 4. 归一化和dropout
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2_2 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerImproved, self).__setstate__(state)

    def forward(self, src, memory2=None, src_mask=None, src_key_padding_mask=None):
        # 子层1: 自注意力 (带残差和层归一化)
        src1 = self.norm1(src)
        src2 = self.self_attn(src1, src1, src1, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) # 残差连接
        # 可选的子层2: 处理第二全局输入
        if memory2 is not None:
            src2_2 = self.linear_global2(memory2) # 投影到d_model维度
            src = src + self.dropout2_2(src2_2)
        # 子层3: 前馈网络 (带残差和层归一化)
        src1 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src1))))
        src = src + self.dropout2(src2)
        return src

from torch.nn import MultiheadAttention as nnMultiheadAttention
#通用解码器：标准解码器层，支持自注意力和交叉注意力
class TransformerDecoderLayerImproved(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayerImproved, self).__init__()
        # 1. 自注意力层 (处理目标序列内部依赖)
        self.self_attn = nnMultiheadAttention(d_model, nhead, dropout=dropout)
        #self.self_attn = MultiheadAttention(d_model,d_model, nhead, dropout=dropout)
        # 2. 交叉注意力层 (连接编码器和解码器)
        self.multihead_attn = nnMultiheadAttention(d_model, nhead, dropout=dropout)
        # 3. 前馈网络Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        # 4. 三个归一化层
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerImproved, self).__setstate__(state)

    def forward(self, tgt, memory, memory2=None,tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 子层1: 自注意力 (因果掩码防止信息泄漏)
        tgt1 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # 子层2: 交叉注意力 (查询来自解码器，键/值来自编码器)
        tgt1 = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt1, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        #print("cross-attention")
        tgt = tgt + self.dropout2(tgt2)
        # 子层3: 前馈网络
        tgt1 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt1))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

#CAD命令/参数解码器：简化解码器层，使用线性投影而非交叉注意力
class TransformerDecoderLayerGlobalImproved(Module):
    def __init__(self, d_model, d_global, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", d_global2=None):
        super(TransformerDecoderLayerGlobalImproved, self).__init__()
        self.self_attn = MultiheadAttention(d_model,d_model, nhead, dropout=dropout)
        # 2. 全局特征线性投影 (替代交叉注意力)
        self.linear_global = Linear(d_global, d_model)

        if d_global2 is not None:
            self.linear_global2 = Linear(d_global2, d_model)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout2_2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayerGlobalImproved, self).__setstate__(state)

    def forward(self, tgt, memory, memory2=None, tgt_mask=None, tgt_key_padding_mask=None, *args, **kwargs):
        tgt1 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt1, tgt1, tgt1, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # 子层2: 全局特征投影 (简化版的交叉注意力)
        tgt2 = self.linear_global(memory)  # memory是全局特征z
        tgt = tgt + self.dropout2(tgt2)  # 直接相加，隐式广播implicit broadcast

        if memory2 is not None:
            tgt2_2 = self.linear_global2(memory2)
            tgt = tgt + self.dropout2_2(tgt2_2)

        tgt1 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt1))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt




from model.mamba_ssm.modules.mamba_simple import Mamba
import torch.nn as nn

class MambaEncoderLayer(nn.Module):
    def __init__(self,d_model,dropout=0.1,d_state=16,d_conv=4,expand=2,):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src,src_mask=None,src_key_padding_mask=None,**kwargs ):
        """
        src: (S, B, D)  ← 和 TransformerEncoderLayer 完全一致
        """
        residual = src
        x = self.norm(src)
        # Mamba 需要 (B, S, D)
        x = x.transpose(0, 1)
        x = self.mamba(x)
        x = x.transpose(0, 1)
        return residual + self.dropout(x)


class MambaDecoderLayerGlobalImproved(nn.Module):
    def __init__(
        self,
        d_model,
        d_global,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        d_state=16,
        d_conv=4,
        expand=2,
        d_global2=None,
    ):
        super().__init__()

        # 1替代 self-attention
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # 2Global z（完全保留）
        self.linear_global = nn.Linear(d_global, d_model)
        if d_global2 is not None:
            self.linear_global2 = nn.Linear(d_global2, d_model)
        # 3️FFN（不动）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
    def forward(self,tgt,memory,memory2=None,tgt_mask=None,tgt_key_padding_mask=None,**kwargs):
        """
        tgt:     (S, B, D)
        memory:  (1, B, d_global)
        """
        # ---- Mamba self modeling ----
        residual = tgt
        x = self.norm1(tgt)

        x = x.transpose(0, 1)     # (B, S, D)
        x = self.mamba(x)
        x = x.transpose(0, 1)

        tgt = residual + self.dropout(x)

        # ---- Global conditioning（完全保留）----
        tgt = tgt + self.linear_global(memory)
        if memory2 is not None:
            tgt = tgt + self.linear_global2(memory2)

        # ---- FFN ----
        residual = tgt
        x = self.norm2(tgt)
        x = self.linear2(self.dropout_ffn(self.activation(self.linear1(x))))
        tgt = residual + x

        return tgt
