"""Caption decoders over frozen ResNet-50 grid features.

Every decoder exposes the same decoding interface used by src/decoding.py:
    state = model.init_state(features)            # features: [B, 49, 2048]
    log_probs, state = model.step(tokens, state)  # tokens: [B, t] prefix -> [B, vocab] next-token log-probs
All tensors in `state` have batch as their first dimension, so beam search can reorder them.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import PAD_IDX


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]

    def forward(self, query, key_value, causal=False, need_weights=False):
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))
        dropout_p = self.dropout if self.training else 0.0

        if need_weights:
            # Explicit attention so the weights can be returned (visualization only).
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
            if causal:
                Tq, Tk = scores.shape[-2:]
                mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            weights = scores.softmax(dim=-1)
            out = F.dropout(weights, dropout_p) @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)

        B, _, Tq, _ = out.shape
        out = out.transpose(1, 2).reshape(B, Tq, -1)
        return self.out_proj(out), weights


class DecoderLayer(nn.Module):
    """Pre-LayerNorm decoder block: masked self-attention -> cross-attention over image grid -> FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, need_weights=False):
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, causal=True)
        x = x + self.drop(h)
        h, cross_weights = self.cross_attn(self.norm2(x), memory, need_weights=need_weights)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, cross_weights


class GridCaptioner(nn.Module):
    """Shared image side: each grid cell -> Linear -> LayerNorm, plus a learned 2D (row + column) position.

    Both decoders use exactly this input, so the Transformer vs LSTM ablation compares decoders only.
    """

    def _build_image_side(self, feat_dim, grid_size, d_model):
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.feat_norm = nn.LayerNorm(d_model)
        self.row_embed = nn.Parameter(torch.zeros(grid_size, 1, d_model))
        self.col_embed = nn.Parameter(torch.zeros(1, grid_size, d_model))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def encode(self, features):
        """[B, 49, feat_dim] -> memory [B, 49, d_model]."""
        pos = (self.row_embed + self.col_embed).flatten(0, 1)
        x = self.feat_norm(self.feat_proj(features.to(self.feat_proj.weight.dtype))) + pos
        return self.drop(x)


class TransformerCaptioner(GridCaptioner):
    def __init__(
        self,
        vocab_size,
        feat_dim=2048,
        grid_size=7,
        d_model=512,
        n_heads=8,
        n_layers=3,
        d_ff=2048,
        dropout=0.1,
        max_len=64,
    ):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, feat_dim=feat_dim, grid_size=grid_size, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout, max_len=max_len,
        )
        self._build_image_side(feat_dim, grid_size, d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)
        # Output projection is tied to tok_embed (see `decode`); only a bias is separate.
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))

        self.apply(self._init_weights)
        for p in (self.row_embed, self.col_embed, self.pos_embed):
            nn.init.normal_(p, std=0.02)

    def decode(self, tokens, memory, need_weights=False, last_only=False):
        """tokens [B, T] -> logits [B, T, vocab]; optionally per-layer cross-attention [B, H, T, 49].

        last_only=True projects only the final position to the vocabulary (logits [B, 1, vocab]); decoding
        needs nothing else, and full-prefix logits for thousands of beams would not fit in GPU memory.
        """
        T = tokens.size(1)
        x = self.drop(self.tok_embed(tokens) + self.pos_embed[:T])
        cross_weights = []
        for layer in self.layers:
            x, w = layer(x, memory, need_weights=need_weights)
            cross_weights.append(w)
        x = x[:, -1:] if last_only else x
        logits = F.linear(self.final_norm(x), self.tok_embed.weight, self.out_bias)
        return (logits, cross_weights) if need_weights else logits

    def forward(self, features, tokens):
        return self.decode(tokens, self.encode(features))

    def attention_maps(self, features, tokens):
        """Cross-attention used to predict each next token, averaged over layers and heads: [B, T, 49]."""
        _, weights = self.decode(tokens, self.encode(features), need_weights=True)
        return torch.stack(weights).mean(dim=(0, 2))

    # Decoding interface ------------------------------------------------------
    def init_state(self, features):
        return {"memory": self.encode(features)}

    def step(self, tokens, state):
        # No KV cache: the full prefix is re-run each step. Captions are short (<~30 tokens),
        # so this stays fast and keeps the attention code simple.
        logits = self.decode(tokens, state["memory"], last_only=True)[:, -1]
        return logits.float().log_softmax(dim=-1), state


class LSTMAttentionCaptioner(GridCaptioner):
    """Show, Attend and Tell style decoder: an LSTM that attends over the 49 grid cells at every step.

    Step t: additive (Bahdanau) attention with h_{t-1} -> gated context z_t -> LSTMCell([word_{t-1}; z_t])
    -> deep output W[h_t; z_t] -> logits via the tied word embedding.
    """

    def __init__(self, vocab_size, feat_dim=2048, grid_size=7, d_model=512, hidden_size=1024, attn_dim=512,
                 dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, feat_dim=feat_dim, grid_size=grid_size, d_model=d_model,
                           hidden_size=hidden_size, attn_dim=attn_dim, dropout=dropout)
        self._build_image_side(feat_dim, grid_size, d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.init_h = nn.Linear(d_model, hidden_size)
        self.init_c = nn.Linear(d_model, hidden_size)
        self.attn_mem = nn.Linear(d_model, attn_dim)
        self.attn_hid = nn.Linear(hidden_size, attn_dim)
        self.attn_score = nn.Linear(attn_dim, 1)
        self.gate = nn.Linear(hidden_size, d_model)  # "beta" gate: how much image context to let in
        self.lstm = nn.LSTMCell(2 * d_model, hidden_size)
        self.out_proj = nn.Linear(hidden_size + d_model, d_model)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights)
        nn.init.normal_(self.row_embed, std=0.02)
        nn.init.normal_(self.col_embed, std=0.02)

    def init_state(self, features):
        memory = self.encode(features)
        mean = memory.mean(dim=1)
        return {
            "memory": memory,
            "mem_keys": self.attn_mem(memory),  # precomputed once per image
            "h": torch.tanh(self.init_h(mean)),
            "c": torch.tanh(self.init_c(mean)),
        }

    def _step(self, token, state):
        scores = self.attn_score(torch.tanh(state["mem_keys"] + self.attn_hid(state["h"])[:, None])).squeeze(-1)
        alpha = scores.float().softmax(dim=-1)  # [B, 49]
        context = (alpha[..., None].to(state["memory"].dtype) * state["memory"]).sum(dim=1)
        context = torch.sigmoid(self.gate(state["h"])) * context

        word = self.drop(self.tok_embed(token))
        h, c = self.lstm(torch.cat([word, context.to(word.dtype)], dim=-1), (state["h"], state["c"]))
        out = self.out_proj(self.drop(torch.cat([h, context.to(h.dtype)], dim=-1)))
        logits = F.linear(out, self.tok_embed.weight, self.out_bias)
        return logits, {**state, "h": h, "c": c}, alpha

    def forward(self, features, tokens):
        state = self.init_state(features)
        logits = []
        for t in range(tokens.size(1)):
            step_logits, state, _ = self._step(tokens[:, t], state)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    def attention_maps(self, features, tokens):
        """Attention used to predict each next token: [B, T, 49]."""
        state = self.init_state(features)
        maps = []
        for t in range(tokens.size(1)):
            _, state, alpha = self._step(tokens[:, t], state)
            maps.append(alpha)
        return torch.stack(maps, dim=1)

    # Decoding interface ------------------------------------------------------
    def step(self, tokens, state):
        logits, state, _ = self._step(tokens[:, -1], state)
        return logits.float().log_softmax(dim=-1), state


MODEL_CLASSES = {"TransformerCaptioner": TransformerCaptioner, "LSTMAttentionCaptioner": LSTMAttentionCaptioner}


def load_checkpoint(path, device):
    """Rebuild a model from a checkpoint saved by src.train.save_checkpoint. Returns (model in eval mode, checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MODEL_CLASSES[ckpt["model_class"]](**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
