"""Caption tokenization, word-level vocabulary, and datasets over precomputed grid features."""
import html
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = range(4)

FEATURES_FILE = "features_resnet50_7x7_fp16.npy"
CAPTIONS_FILE = "captions.json"
VOCAB_FILE = "vocab.json"


def tokenize(caption):
    """Lowercase, drop apostrophes ("man's" -> "mans"), split on anything non-alphanumeric.

    Matches the token convention of Karpathy's Flickr30k preprocessing.
    """
    text = re.sub(r"['’`]", "", html.unescape(caption).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).split()


class Vocabulary:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        assert self.itos[:4] == SPECIALS

    @classmethod
    def build(cls, token_lists, min_freq=5):
        counts = Counter(t for tokens in token_lists for t in tokens)
        words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls(SPECIALS + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [SOS_IDX] + [self.stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]

    def decode(self, ids):
        words = []
        for i in ids:
            i = int(i)
            if i == EOS_IDX:
                break
            if i not in (PAD_IDX, SOS_IDX):
                words.append(self.itos[i])
        return " ".join(words)

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))


def load_processed(processed_dir):
    """Returns (features memmap [N, 49, 2048] float16, image records, vocab).

    Each image record: {"filename", "split", "feat_idx", "captions": [5 raw strings]}.
    """
    processed_dir = Path(processed_dir)
    features = np.load(processed_dir / FEATURES_FILE, mmap_mode="r")
    images = json.loads((processed_dir / CAPTIONS_FILE).read_text())
    vocab = Vocabulary.load(processed_dir / VOCAB_FILE)
    return features, images, vocab


def split_images(images, split):
    return [img for img in images if img["split"] == split]


def reference_captions(images):
    """{filename: [tokenized caption strings]} for fast in-training metrics."""
    return {img["filename"]: [" ".join(tokenize(c)) for c in img["captions"]] for img in images}


class CaptionDataset(Dataset):
    """One item per (image, caption) pair: (grid features [49, 2048], encoded caption)."""

    def __init__(self, features, images, vocab, max_words=None):
        self.features = features
        self.samples = [
            (img["feat_idx"], vocab.encode(tokenize(c)[:max_words]))
            for img in images
            for c in img["captions"]
        ]

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [len(ids) for _, ids in self.samples]

    def __getitem__(self, i):
        feat_idx, ids = self.samples[i]
        return torch.from_numpy(np.array(self.features[feat_idx])), torch.tensor(ids)


class LengthBucketSampler(Sampler):
    """Batches of similar caption length, in random order every epoch.

    Indices are shuffled, cut into chunks of `bucket_batches` batches, sorted by length inside each
    chunk and split into batches. Padding (and so the number of LSTM steps per batch) drops a lot,
    while batch composition stays random at the chunk level.
    """

    def __init__(self, lengths, batch_size, bucket_batches=50, drop_last=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.chunk = batch_size * bucket_batches
        self.drop_last = drop_last

    def _batches(self):
        perm = torch.randperm(len(self.lengths)).tolist()
        batches = []
        for s in range(0, len(perm), self.chunk):
            idx = sorted(perm[s : s + self.chunk], key=self.lengths.__getitem__)
            batches += [idx[j : j + self.batch_size] for j in range(0, len(idx), self.batch_size)]
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        return batches

    def __iter__(self):
        batches = self._batches()
        for i in torch.randperm(len(batches)).tolist():
            yield batches[i]

    def __len__(self):
        n = len(self.lengths) // self.batch_size
        return n if self.drop_last else math.ceil(len(self.lengths) / self.batch_size)


def collate_captions(batch):
    feats, caps = zip(*batch)
    return torch.stack(feats), pad_sequence(caps, batch_first=True, padding_value=PAD_IDX)


class ImageFeatureDataset(Dataset):
    """One item per image: (grid features [49, 2048], position in `images`). Used for decoding."""

    def __init__(self, features, images):
        self.features = features
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.features[self.images[i]["feat_idx"]])), i
