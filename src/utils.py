"""Shared helpers: seeding, ResNet-50 grid feature extraction, fast caption metrics."""
import contextlib
import io
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

RESNET_WEIGHTS = torchvision.models.ResNet50_Weights.IMAGENET1K_V1

# Squash to 224x224 instead of resize + center crop: a crop can cut off objects the captions mention.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform=IMAGE_TRANSFORM):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB"))


def load_resnet50(device):
    return torchvision.models.resnet50(weights=RESNET_WEIGHTS).to(device).eval()


@torch.no_grad()
def extract_grid_features(resnet, images):
    """[B, 3, 224, 224] -> [B, 49, 2048]: ResNet-50 up to its last conv block (before avgpool / fc)."""
    x = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(images))))
    x = resnet.layer4(resnet.layer3(resnet.layer2(resnet.layer1(x))))
    return x.flatten(2).transpose(1, 2)


def caption_scores(references, hypotheses):
    """BLEU-1..4 and CIDEr over already-tokenized strings (fast, no Java; used during training).

    references: {id: [caption, ...]}, hypotheses: {id: caption}. The full paper-comparable
    evaluation (PTB tokenizer, METEOR, ROUGE-L) lives in notebook 04.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    gts = {i: references[i] for i in hypotheses}
    res = {i: [h] for i, h in hypotheses.items()}
    with contextlib.redirect_stdout(io.StringIO()):  # Bleu prints its internals
        bleu, _ = Bleu(4).compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    # Plain floats: numpy scalars would end up pickled inside checkpoints.
    return {**{f"BLEU-{n + 1}": float(b) for n, b in enumerate(bleu)}, "CIDEr": float(cider)}


def ensure_java():
    """METEOR and the PTB tokenizer shell out to `java`; put a Temurin install on PATH if it isn't already."""
    if shutil.which("java"):
        return shutil.which("java")
    for bin_dir in sorted(Path("C:/Program Files/Eclipse Adoptium").glob("*/bin"), reverse=True):
        if (bin_dir / "java.exe").exists():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
            return str(bin_dir / "java.exe")
    raise RuntimeError("Java not found: install a JRE (e.g. Temurin 21) for METEOR and the PTB tokenizer")


def official_scores(references, hypotheses, per_image=False):
    """Paper-comparable metrics via pycocoevalcap: PTB tokenizer, BLEU-1..4, METEOR, ROUGE-L, CIDEr.

    references: {id: [raw caption, ...]}, hypotheses: {id: caption}. With per_image=True also returns
    {id: CIDEr} for per-image analysis.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ensure_java()
    tokenizer = PTBTokenizer()
    ids = list(hypotheses)
    gts = tokenizer.tokenize({i: [{"caption": c} for c in references[i]] for i in ids})
    res = tokenizer.tokenize({i: [{"caption": hypotheses[i]}] for i in ids})
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = Bleu(4).compute_score(gts, res)
    meteor, _ = Meteor().compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, cider_per_image = Cider().compute_score(gts, res)
    scores = {**{f"BLEU-{n + 1}": float(b) for n, b in enumerate(bleu)},
              "METEOR": float(meteor), "ROUGE-L": float(rouge), "CIDEr": float(cider)}
    if per_image:
        return scores, dict(zip(ids, map(float, cider_per_image)))
    return scores
