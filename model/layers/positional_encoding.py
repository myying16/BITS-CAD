import math
import torch
import torch.nn as nn

#基于正弦和余弦函数的固定编码
# PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
class PositionalEncodingSinCos(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingSinCos, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #调整维度：(max_len, d_model) → (max_len, 1, d_model) → (1, max_len, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) #注册为缓冲区（不参与训练）
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#基于查找表的可学习编码
class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):  #[256, 100, 256]
        pos = self.position[:x.size(0)]
        #print("x",x.shape)
        #print("pos",pos.shape)
        x = x + self.pos_embed(pos)
        return self.dropout(x)
