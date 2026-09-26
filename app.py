"""Web demo: caption an image with any of the trained decoders and see where the model looked.

    python app.py                 # http://127.0.0.1:8000
    python app.py --port 9000     # another port

The front-end lives in web/; this module is the API it talks to.
"""
import argparse
import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.dataset import SOS_IDX, Vocabulary
from src.decoding import beam_search, greedy_decode
from src.models import load_checkpoint
from src.utils import IMAGE_TRANSFORM, extract_grid_features, load_resnet50

ROOT = Path(__file__).parent
WEB = ROOT / "web"
CKPT = ROOT / "checkpoints"
EVAL = ROOT / "results" / "evaluation"
IMAGES = ROOT / "data" / "raw" / "Images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DECODERS = {"transformer": "Transformer", "lstm": "LSTM + attention"}
TRAININGS = {"xe": "cross-entropy", "scst": "+ SCST"}
MAX_WORDS = 30
SAMPLES = ["7130336193.jpg", "150411291.jpg", "2468466969.jpg", "771048251.jpg", "441817653.jpg", "2738077433.jpg"]


@lru_cache(maxsize=1)
def encoder():
    return load_resnet50(DEVICE)


@lru_cache(maxsize=1)
def vocabulary():
    return Vocabulary.load(ROOT / "data" / "processed" / "vocab.json")


@lru_cache(maxsize=4)
def captioner(checkpoint):
    return load_checkpoint(CKPT / checkpoint, DEVICE)[0]


@lru_cache(maxsize=1)
def catalogue():
    """Every trained checkpoint, with the beam settings tuned on val and its test CIDEr."""
    try:
        tuned = json.loads((EVAL / "decoding_choice.json").read_text())
    except FileNotFoundError:
        tuned = {}
    try:
        metrics = pd.read_csv(EVAL / "test_metrics.csv")
    except FileNotFoundError:
        metrics = pd.DataFrame(columns=["decoder", "training", "CIDEr"])

    models = []
    for decoder, decoder_label in DECODERS.items():
        for training, training_label in TRAININGS.items():
            if not (CKPT / f"{decoder}_{training}.pt").exists():
                continue
            rows = metrics[(metrics.decoder == decoder_label) & (metrics.training == training.upper())]
            beam = tuned.get(f"{decoder_label} · {training.upper()}", {"beam_size": 5, "length_penalty": 1.0})
            models.append({
                "decoder": decoder, "training": training,
                "label": f"{decoder_label} · {training_label}",
                "beam_size": int(beam["beam_size"]), "length_penalty": float(beam["length_penalty"]),
                "test_cider": round(float(rows.CIDEr.max()), 1) if len(rows) else None,
            })
    return models


def encode_jpeg(image, quality=88):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def attention_tiles(image, words, attn, size=224):
    """One tile per generated word: the image stays bright where the decoder attended."""
    base = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255
    tiles = []
    for word, weights in zip(words, attn):
        a = np.asarray(weights, dtype=np.float32).reshape(7, 7)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        mask = np.asarray(Image.fromarray(np.uint8(255 * a)).resize((size, size), Image.BICUBIC), dtype=np.float32)
        spotlight = base * (0.24 + 0.76 * mask[..., None] / 255)
        tiles.append({"word": word, "image": encode_jpeg(Image.fromarray(np.uint8(255 * spotlight)))})
    return tiles


@torch.no_grad()
def run_caption(image, decoder, training, strategy, beam_size, length_penalty):
    checkpoint = f"{decoder}_{training}.pt"
    if not (CKPT / checkpoint).exists():
        raise HTTPException(404, f"{checkpoint} has not been trained yet")

    model, vocab = captioner(checkpoint), vocabulary()
    pixels = IMAGE_TRANSFORM(image.convert("RGB"))[None].to(DEVICE)
    features = extract_grid_features(encoder(), pixels)

    if strategy == "greedy":
        ids = greedy_decode(model, features, max_len=MAX_WORDS)[0]
        setting = "greedy"
    else:
        ids = beam_search(model, features, beam_size=beam_size, length_penalty=length_penalty, max_len=MAX_WORDS)[0]
        setting = f"beam k={beam_size}, α={length_penalty:g}"

    tokens = torch.tensor([[SOS_IDX] + ids], device=DEVICE)
    attn = model.attention_maps(features, tokens)[0].float().cpu().numpy()
    words = [vocab.itos[i] for i in ids] + ["<eos>"]
    model_info = next(m for m in catalogue() if m["decoder"] == decoder and m["training"] == training)
    return {
        "caption": vocab.decode(ids),
        "setting": setting,
        "model": model_info["label"],
        "test_cider": model_info["test_cider"],
        "words": len(ids),
        "tiles": attention_tiles(image, words, attn),
    }


app = FastAPI(title="Image captioning · Flickr30k")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/models")
def models():
    return {"models": catalogue(), "device": str(DEVICE),
            "samples": [s for s in SAMPLES if (IMAGES / s).exists()]}


@app.get("/api/sample/{name}")
def sample(name: str):
    path = IMAGES / Path(name).name
    if not path.exists():
        raise HTTPException(404, "sample not found")
    return FileResponse(path)


@app.post("/api/caption")
async def caption(
    file: UploadFile | None = File(None),
    sample_name: str | None = Form(None),
    decoder: str = Form("transformer"),
    training: str = Form("scst"),
    strategy: str = Form("beam"),
    beam_size: int = Form(5),
    length_penalty: float = Form(1.0),
):
    if file is not None:
        image = Image.open(io.BytesIO(await file.read()))
    elif sample_name:
        image = Image.open(IMAGES / Path(sample_name).name)
    else:
        raise HTTPException(400, "send an image file or a sample name")
    return run_caption(image, decoder, training, strategy, max(1, beam_size), length_penalty)


app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"device: {DEVICE} · http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
