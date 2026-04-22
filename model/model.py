from .layers.mamba_encoder import EncoderMambaBiBlock
from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask_svg, _get_key_padding_mask_svg
from model.text.text_embed import TextEmbedder
from model.text.adaptive_layer import TextAdaptiveLayer

class SVGEmbedding(nn.Module):
    """Embedding: command embed + parameter embed + positional embed"""
    def __init__(self, cfg, seq_len):
        super().__init__()
        """concatenation-based"""
        self.command_embed = nn.Embedding(cfg.svg_n_commands, 12)
        args_dim = cfg.args_dim + 1  # 256+1=257
        self.args_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.args_mlp = nn.Linear(64 * cfg.svg_n_args, 128)
        self.mlp = nn.Linear(4 + 8 + 128, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len + 2)
    def forward(self, command, args):
        S, N = command.shape
        command_embedding = self.command_embed(command.long())
        args_embedding = self.args_mlp(self.args_embed((args + 1).long()).view(S, N, -1))
        """concatenation-based"""
        src = torch.cat([command_embedding, args_embedding], dim=-1)
        src = self.mlp(src)
        src = self.pos_encoding(src)
        return src


class TextEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_embedder = TextEmbedder(cfg.text_model, cfg.text_cache_dir, cfg.text_max_len)
        self.adaptive_layer = TextAdaptiveLayer(cfg.text_embed_dim, cfg.d_model, cfg.n_heads,cfg.dropout)
        for p in self.text_embedder.parameters():
            p.requires_grad = False
        self.text_embedder.eval()
    def forward(self, texts):
        text_embed, text_mask = self.text_embedder.get_embedding(texts)
        mask_prompt_dict = {
            "attn_mask": None,
            "key_padding_mask": text_mask
        }
        text_embedding, _ = self.adaptive_layer(text_embed, mask_prompt_dict)
        valid_mask = (~text_mask).float().unsqueeze(-1)
        text_global = (text_embedding * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        text_embedding = text_global.unsqueeze(0).expand(1, -1, -1)
        return text_embedding



class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.cad_max_total_len)
        self.seq_len = cfg.cad_max_total_len  #60
    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        seq_len = cfg.svg_max_total_len
        self.embedding = SVGEmbedding(cfg, seq_len)
        self.blocks = nn.ModuleList([
            EncoderMambaBiBlock(cfg)
            for _ in range(cfg.n_layers)
        ])
        self.encoder_norm = LayerNorm(cfg.d_model)

    def forward(self, command, args):
        padding_mask, key_padding_mask = _get_padding_mask_svg(command, seq_dim=0), _get_key_padding_mask_svg(command,                                                                                                      seq_dim=0)
        src = self.embedding(command, args)   # (S, N, 256)
        for blk in self.blocks:
            src = blk(src, padding_mask=padding_mask)
        memory = self.encoder_norm(src)
        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True)
        return z


class Fusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, use_gate=True):
        super().__init__()
        self.use_gate = use_gate

        self.affine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        self.dropout = nn.Dropout(dropout)
    def forward(self, H_svg, H_text):
        gamma_beta = self.affine(H_text)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated = (1 + gamma) * H_svg + beta
        if self.use_gate:
            g = self.gate(H_text)
            H_out = H_svg + g * (modulated - H_svg)
        else:
            H_out = modulated
        return self.dropout(H_out)



class CommandFCN(nn.Module):
    def __init__(self, d_model, n_commands):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_commands)
        )
    def forward(self, out):
        command_logits = self.mlp(out)  # Shape [S, N, n_commands]
        return command_logits


class ArgsFCN(nn.Module):
    def __init__(self, d_model, n_args, args_dim=256):
        super().__init__()
        self.n_args = n_args
        self.args_dim = args_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, n_args * args_dim)
        )
    def forward(self, out):
        S, N, _ = out.shape
        args_logits = self.mlp(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]
        return args_logits



class CommandDecoder(nn.Module):
    def __init__(self, cfg):
        super(CommandDecoder, self).__init__()
        self.embedding = ConstEmbedding(cfg)
        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward,
                                                            cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)
        self.fcn = CommandFCN(cfg.d_model, cfg.cad_n_commands)
    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        command_logits = self.fcn(out)
        return command_logits, out


class ArgsDecoder(nn.Module):
    def __init__(self, cfg):
        super(ArgsDecoder, self).__init__()
        self.embedding = ConstEmbedding(cfg)
        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward,
                                                             cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)
        args_dim = cfg.args_dim + 1
        self.fcn = ArgsFCN(cfg.d_model, cfg.cad_n_args, args_dim)
    def forward(self, z, guidance):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)
        # guidance
        out = out + guidance
        args_logits = self.fcn(out)
        return args_logits


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2),
                                        nn.GELU(),
                                        nn.Linear(cfg.d_model // 2, cfg.d_model))
    def forward(self, z):
        return z + self.bottleneck(z)


class SVG2CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(SVG2CADTransformer, self).__init__()
        self.args_dim = cfg.args_dim + 1
        self.encoder = Encoder(cfg)
        self.text_embedding=TextEmbedding(cfg)
        self.feature=Fusion(cfg.d_model, cfg.dropout)
        self.bottleneck = Bottleneck(cfg)

        self.command_decoder = CommandDecoder(cfg)
        self.args_decoder = ArgsDecoder(cfg)
    def forward(self, texts_enc,  commands_enc, args_enc):
        commands_enc_, args_enc_= _make_seq_first( commands_enc,args_enc)  #N, S, ... -> S, N, ...

        z = self.encoder( commands_enc_, args_enc_)
        text_embedding = self.text_embedding(texts_enc)
        z = self.feature( z,text_embedding)
        z = self.bottleneck(z)      #1, 256, 256

        """command-guided generation"""
        command_logits, guidance = self.command_decoder(z)
        command_logits = _make_batch_first(command_logits)  # S, N, ... -> N, S, ...
        args_logits = self.args_decoder(z, guidance)
        args_logits = _make_batch_first(args_logits)
        res = {
            "command_logits": command_logits,
            "args_logits": args_logits
        }
        return res