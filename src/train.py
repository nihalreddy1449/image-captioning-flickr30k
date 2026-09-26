"""Training loop shared by every decoder (Transformer now, LSTM + attention later)."""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from .dataset import PAD_IDX
from .decoding import generate_captions
from .utils import caption_scores, set_seed


def build_optimizer(model, lr, weight_decay):
    """AdamW with weight decay only on weight matrices (not biases, norms, or embeddings)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embed" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98))


def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup to the peak lr, then cosine decay to zero."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _token_loss(model, feats, caps, criterion):
    """Teacher forcing: predict caps[:, 1:] from caps[:, :-1]. Returns (mean loss, #target tokens)."""
    logits = model(feats, caps[:, :-1])
    targets = caps[:, 1:]
    loss = criterion(logits.float().flatten(0, 1), targets.flatten())
    return loss, (targets != PAD_IDX).sum()


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total, n_tokens = torch.zeros((), device=device), torch.zeros((), device=device)
    for feats, caps in loader:
        feats = feats.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, n = _token_loss(model, feats, caps, criterion)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total += loss.detach() * n
        n_tokens += n
    return (total / n_tokens).item()


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Plain token-level cross-entropy (no label smoothing), so it is comparable across runs."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total, n_tokens = torch.zeros((), device=device), torch.zeros((), device=device)
    for feats, caps in loader:
        feats = feats.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, n = _token_loss(model, feats, caps, criterion)
        total += loss * n
        n_tokens += n
    return (total / n_tokens).item()


def save_checkpoint(model, path, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_class": type(model).__name__,
        "model_config": model.config,
        "model_state": model.state_dict(),
        **extra,
    }, path)


def fit(model, train_loader, val_loader, val_image_loader, val_refs, vocab, device, *,
        epochs, lr, weight_decay, warmup_epochs, label_smoothing, grad_clip, ckpt_path, run_config,
        on_epoch_end=None):
    """Train for a fixed cosine schedule; keep the checkpoint with the best greedy val CIDEr.

    on_epoch_end(history) is called after every epoch (e.g. a live notebook table); without it,
    one line per epoch is printed.
    """
    optimizer = build_optimizer(model, lr, weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine_schedule(optimizer, warmup_epochs * steps_per_epoch, epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)

    history, best_cider = [], float("-inf")
    for epoch in range(1, epochs + 1):
        start = time.time()
        lr_now = scheduler.get_last_lr()[0]
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, grad_clip)
        val_loss = evaluate_loss(model, val_loader, device)
        captions = generate_captions(model, val_image_loader, vocab, device)
        scores = caption_scores(val_refs, captions)
        improved = scores["CIDEr"] > best_cider
        history.append({
            "epoch": epoch, "lr_start": lr_now, "train_loss": train_loss, "val_loss": val_loss,
            "val_bleu4": scores["BLEU-4"], "val_cider": scores["CIDEr"], "seconds": time.time() - start,
            "best": improved,
        })
        if improved:
            best_cider = scores["CIDEr"]
            save_checkpoint(model, ckpt_path, run_config=run_config, epoch=epoch, val_scores=scores)

        if on_epoch_end is not None:
            on_epoch_end(history)
        else:
            row = history[-1]
            print(f"epoch {epoch:2d} | lr {lr_now:.2e} | train {train_loss:.3f} | val loss {val_loss:.3f} "
                  f"| BLEU-4 {row['val_bleu4']:.3f} | CIDEr {row['val_cider']:.3f} | {row['seconds']:.0f}s"
                  + ("  *best" if improved else ""))
    return history


def run_experiment(name, build_model, train_fn, ckpt_path, history_path, seed=42):
    """Train one named run unless it already finished (history file written last = completion marker).

    build_model() -> fresh model; train_fn(model) -> history. Returns (history, cached: bool).
    """
    history_path = Path(history_path)
    if history_path.exists() and Path(ckpt_path).exists():
        return json.loads(history_path.read_text()), True
    set_seed(seed)
    history = train_fn(build_model())
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    return history, False
