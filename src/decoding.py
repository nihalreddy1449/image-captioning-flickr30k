"""Greedy and beam-search decoding, written against the model.init_state / model.step interface."""
import torch

from .dataset import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX

# Tokens a generated caption may never contain. Banning <unk> makes the decoder commit to a real word.
BANNED_IDS = [PAD_IDX, SOS_IDX, UNK_IDX]


def _until_eos(ids):
    return ids[: ids.index(EOS_IDX)] if EOS_IDX in ids else ids


@torch.no_grad()
def greedy_decode(model, features, max_len=30):
    """features [B, 49, 2048] -> list of B token-id lists (no <sos>/<eos>)."""
    B, device = features.size(0), features.device
    state = model.init_state(features)
    tokens = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        log_probs, state = model.step(tokens, state)
        log_probs[:, BANNED_IDS] = float("-inf")
        next_tok = log_probs.argmax(dim=-1).masked_fill(finished, EOS_IDX)
        tokens = torch.cat([tokens, next_tok[:, None]], dim=1)
        finished |= next_tok == EOS_IDX
        if finished.all():
            break
    return [_until_eos(seq[1:]) for seq in tokens.tolist()]


@torch.no_grad()
def beam_search(model, features, beam_size=3, max_len=30, length_penalty=0.0):
    """Batched beam search over B images at once.

    Finished hypotheses stay in the beam with a frozen score (they can only extend with <pad> at
    zero cost). The final beams are ranked by  sum log-prob / length ** length_penalty,
    so length_penalty=0 ranks by raw log-probability and 1 by per-token average.
    """
    B, K, device = features.size(0), beam_size, features.device
    state = {k: v.repeat_interleave(K, dim=0) for k, v in model.init_state(features).items()}
    tokens = torch.full((B * K, 1), SOS_IDX, dtype=torch.long, device=device)
    scores = torch.zeros(B, K, device=device)
    scores[:, 1:] = float("-inf")  # all K beams start identical, so expand only the first one
    finished = torch.zeros(B * K, dtype=torch.bool, device=device)
    lengths = torch.zeros(B * K, device=device)
    offsets = torch.arange(B, device=device)[:, None] * K

    for _ in range(max_len):
        log_probs, state = model.step(tokens, state)  # [B*K, V]
        log_probs[:, BANNED_IDS] = float("-inf")
        log_probs[finished] = float("-inf")
        log_probs[finished, PAD_IDX] = 0.0
        V = log_probs.size(-1)

        candidates = (scores.view(-1, 1) + log_probs).view(B, K * V)
        scores, flat_idx = candidates.topk(K, dim=-1)  # [B, K]
        beam_idx = (flat_idx // V + offsets).view(-1)
        next_tok = (flat_idx % V).view(-1)

        tokens = torch.cat([tokens[beam_idx], next_tok[:, None]], dim=1)
        state = {k: v[beam_idx] for k, v in state.items()}
        was_finished = finished[beam_idx]
        lengths = lengths[beam_idx] + (~was_finished).float()
        finished = was_finished | (next_tok == EOS_IDX)
        if finished.all():
            break

    ranked = scores / lengths.view(B, K).clamp(min=1) ** length_penalty
    best = (ranked.argmax(dim=-1) + offsets.squeeze(1)).tolist()
    return [_until_eos(tokens[i, 1:].tolist()) for i in best]


@torch.no_grad()
def generate_captions(model, image_loader, vocab, device, beam_size=1, max_len=30, length_penalty=0.0):
    """Caption every image of an ImageFeatureDataset loader -> {filename: caption}. beam_size=1 is greedy."""
    model.eval()
    images = image_loader.dataset.images
    captions = {}
    for feats, idx in image_loader:
        feats = feats.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if beam_size == 1:
                seqs = greedy_decode(model, feats, max_len)
            else:
                seqs = beam_search(model, feats, beam_size, max_len, length_penalty)
        for i, seq in zip(idx.tolist(), seqs):
            captions[images[i]["filename"]] = vocab.decode(seq)
    return captions
