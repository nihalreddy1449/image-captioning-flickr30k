"""Self-critical sequence training (SCST): fine-tune a captioner to directly maximise CIDEr-D.

Rennie et al. 2017, "Self-critical Sequence Training for Image Captioning", with the baseline from
Luo 2020, "A Better Variant of Self-Critical Sequence Training": sample k captions per image and use the
mean reward of the *other* k-1 samples as each sample's baseline (no extra greedy decode needed).
"""
import math
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

from .dataset import EOS_IDX, PAD_IDX, SOS_IDX, tokenize
from .decoding import BANNED_IDS, generate_captions
from .train import save_checkpoint
from .utils import caption_scores

EOS_WORD = "<eos>"


def _ngram_counts(words, n_max=4):
    counts = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            counts[tuple(words[i : i + n])] += 1
    return counts


class CiderD:
    """CIDEr-D with document frequencies fixed from a reference corpus (the training captions).

    Same formula as pycocoevalcap's scorer: tf-idf vectors of 1-4-grams, clipped matches, cosine
    similarity per n, Gaussian length penalty (sigma=6), averaged over n and references, x10.
    pycocoevalcap computes document frequencies from whatever set is being scored; for a reward
    they must be fixed, so they come from all training images here.
    """

    def __init__(self, refs_by_image, sigma=6.0):
        self.sigma = sigma
        self.refs_by_image = refs_by_image
        self.df = Counter()
        for refs in refs_by_image.values():
            self.df.update({ng for words in refs for ng in _ngram_counts(words)})
        self.log_n_docs = math.log(len(refs_by_image))

    def _vec(self, words):
        vec, norm = [defaultdict(float) for _ in range(4)], [0.0] * 4
        length = 0
        for ngram, tf in _ngram_counts(words).items():
            n = len(ngram) - 1
            vec[n][ngram] = tf * (self.log_n_docs - math.log(max(1.0, self.df[ngram])))
            norm[n] += vec[n][ngram] ** 2
            if n == 1:  # pycocoevalcap measures length in bigrams
                length += tf
        return vec, [math.sqrt(x) for x in norm], length

    def score(self, words, image_id):
        return self.score_group([words], image_id)[0]

    def score_group(self, word_lists, image_id):
        """Score several hypotheses for one image, computing the reference vectors once."""
        refs = [self._vec(ref) for ref in self.refs_by_image[image_id]]
        scores = []
        for words in word_lists:
            vec_h, norm_h, len_h = self._vec(words)
            total = 0.0
            for vec_r, norm_r, len_r in refs:
                penalty = math.exp(-((len_h - len_r) ** 2) / (2 * self.sigma**2))
                for n in range(4):
                    val = sum(min(w, vec_r[n][g]) * vec_r[n][g] for g, w in vec_h[n].items() if g in vec_r[n])
                    if norm_h[n] and norm_r[n]:
                        val /= norm_h[n] * norm_r[n]
                    total += val * penalty
            scores.append(total / 4 / len(refs) * 10.0)
        return scores


@torch.no_grad()
def sample_captions(model, features, n_samples, max_len=20):
    """Multinomial sampling of n_samples captions per image -> tokens [B*n, T] (no <sos>, <pad> after <eos>)."""
    feats = features.repeat_interleave(n_samples, dim=0)
    state = model.init_state(feats)
    tokens = torch.full((feats.size(0), 1), SOS_IDX, dtype=torch.long, device=feats.device)
    finished = torch.zeros(feats.size(0), dtype=torch.bool, device=feats.device)
    for _ in range(max_len):
        log_probs, state = model.step(tokens, state)
        log_probs[:, BANNED_IDS] = float("-inf")
        next_tok = torch.multinomial(log_probs.exp(), 1).squeeze(1).masked_fill(finished, PAD_IDX)
        tokens = torch.cat([tokens, next_tok[:, None]], dim=1)
        finished |= next_tok == EOS_IDX
        if finished.all():
            break
    return tokens[:, 1:]


def sequence_log_probs(model, features, samples):
    """Sum of log-probabilities of each sampled caption (tokens up to and including <eos>), with gradients."""
    inputs = torch.cat([torch.full_like(samples[:, :1], SOS_IDX), samples[:, :-1]], dim=1)
    with torch.autocast(features.device.type, dtype=torch.bfloat16):
        logits = model(features, inputs)
    banned = torch.tensor(BANNED_IDS, device=logits.device)
    logits = logits.float().index_fill(-1, banned, float("-inf"))
    mask = samples != PAD_IDX
    token_logp = logits.log_softmax(dim=-1).gather(-1, samples[..., None]).squeeze(-1)
    return token_logp.masked_fill(~mask, 0.0).sum(dim=1), mask


def _to_words(ids, vocab):
    """Token ids -> words, with a final <eos> word only if the caption actually ended."""
    words = []
    for i in ids:
        if i == EOS_IDX:
            return words + [EOS_WORD]
        if i != PAD_IDX:
            words.append(vocab.itos[i])
    return words


def scst_fit(model, features, train_images, vocab, device, val_image_loader, val_refs, *,
             epochs, lr, ckpt_path, run_config, batch_images=50, n_samples=5, max_len=20, grad_clip=1.0,
             on_epoch_end=None):
    """SCST fine-tuning; keeps the checkpoint with the best greedy val CIDEr (epoch 0 = starting model).

    The reward sees an explicit <eos> word on both references and samples, so captions that stop
    mid-phrase ("a man in a") lose the end-of-caption n-grams and are not rewarded.
    """
    cider = CiderD({img["filename"]: [tokenize(c) + [EOS_WORD] for c in img["captions"]] for img in train_images})
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def validate():
        caps = generate_captions(model, val_image_loader, vocab, device)
        return caption_scores(val_refs, caps)

    scores = validate()
    history = [{"epoch": 0, "train_reward": float("nan"), "sample_len": float("nan"), "val_bleu4": scores["BLEU-4"],
                "val_cider": scores["CIDEr"], "seconds": 0.0, "best": True}]
    best_cider = scores["CIDEr"]
    save_checkpoint(model, ckpt_path, run_config=run_config, epoch=0, val_scores=scores)
    if on_epoch_end is not None:
        on_epoch_end(history)

    for epoch in range(1, epochs + 1):
        start = time.time()
        rewards_sum, lengths_sum, n_seen = 0.0, 0.0, 0
        perm = np.random.permutation(len(train_images))
        for s in range(0, len(perm) - batch_images + 1, batch_images):
            batch = [train_images[i] for i in perm[s : s + batch_images]]
            rows = np.array([img["feat_idx"] for img in batch])
            order = np.argsort(rows)  # sorted reads from the memmap are faster
            feats_np = np.empty((len(rows), *features.shape[1:]), dtype=np.float32)
            feats_np[order] = features[rows[order]]
            feats = torch.from_numpy(feats_np).to(device, non_blocking=True)

            model.eval()  # no dropout: sample and score under the same policy
            with torch.autocast(device.type, dtype=torch.bfloat16):
                samples = sample_captions(model, feats, n_samples, max_len)

            sample_lists = samples.tolist()
            rewards = []
            for b, img in enumerate(batch):
                group = [_to_words(sample_lists[b * n_samples + k], vocab) for k in range(n_samples)]
                rewards.extend(cider.score_group(group, img["filename"]))
            rewards = torch.tensor(rewards, device=device).view(len(batch), n_samples)
            baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (n_samples - 1)
            advantage = (rewards - baseline).view(-1)

            seq_logp, mask = sequence_log_probs(model, feats.repeat_interleave(n_samples, dim=0), samples)
            loss = -(advantage * seq_logp).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            rewards_sum += rewards.sum().item()
            lengths_sum += mask.sum().item()
            n_seen += rewards.numel()

        scores = validate()
        improved = scores["CIDEr"] > best_cider
        history.append({
            "epoch": epoch, "train_reward": rewards_sum / n_seen, "sample_len": lengths_sum / n_seen - 1,
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
            print(f"epoch {epoch} | reward {row['train_reward']:.3f} | len {row['sample_len']:.1f} | "
                  f"val BLEU-4 {row['val_bleu4']:.3f} | val CIDEr {row['val_cider']:.3f} | {row['seconds']:.0f}s"
                  + ("  *best" if improved else ""))
    return history
