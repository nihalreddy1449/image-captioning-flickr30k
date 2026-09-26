# Image Captioning Project — Context for Claude Code

## Who this is for
Nihal Reddy KV, B.Tech CSE (AI) student at Amrita Vishwa Vidyapeetham. This is a
resume-building project, built end-to-end with real engineering depth rather than
a quick tutorial-level result. He has an RTX 5050 GPU with 8GB VRAM available locally.
This project follows the same build philosophy as his other resume project (a RAG
system, `ms-annual-report-rag`): built from scratch where it adds resume value,
structured like a proper GitHub repo, with incremental commits.

## Working style
Decisions in this project were made one at a time: for each open question, options
were laid out with pros/cons and a recommendation, and Nihal picked. Continue that
pattern for any remaining open decisions during implementation — don't silently
choose defaults for anything architecturally significant; surface the tradeoff and
let him decide, the way this document's decisions were made.

## Project goal
Build an image captioning system: given an input image, generate a natural-language
caption. The project is explicitly designed around an **ablation study** (decoder
architecture and decoding strategy comparisons) as the resume differentiator, rather
than just producing one working model.

## Architecture decisions

### Overall approach: Hybrid
- Frozen pretrained encoder for image features (not trained).
- Custom decoder trained from scratch (this is the part with real engineering depth).
- Rejected alternatives: fully-from-scratch encoder+decoder (infeasible to get good
  results on 8GB VRAM with a small dataset), and fine-tuning an off-the-shelf
  vision-language model like BLIP (too little depth/differentiation for the resume).

### Encoder: ResNet-50 (ImageNet-pretrained, frozen)
- Extract spatial grid features (7×7×2048) for attention.
- Matches the classic "Show, Attend and Tell" architecture — well-documented,
  reliable baseline, fast to get the pipeline working.
- Considered but rejected: ViT (patch tokens, more modern but less of a debuggable
  baseline), CLIP encoder (strong semantics but fiddlier to extract patch-level
  features from). CLIP is a candidate for a later encoder-comparison extension,
  not part of the initial build.

### Decoder: Transformer decoder, built FIRST
- Build order deliberately chosen: Transformer decoder first (chosen over building
  LSTM+Attention first, which was the initially recommended safer option — Nihal
  chose to lead with the Transformer decoder instead).
- LSTM + Attention decoder to be added afterward as the ablation comparison.
- Additional planned ablation axis: greedy decoding vs. beam search at inference.

### Vocabulary / tokenization: word-level, built from scratch
- Vocabulary built directly from Flickr30k training captions (not a pretrained
  subword tokenizer, not pretrained embeddings like GloVe).
- Rationale: keeps the "decoder trained from scratch" story clean, and Flickr30k's
  caption vocabulary (~8–10k unique words) is small enough that this is fully
  tractable without needing subword handling.
- Standard special tokens: `<sos>`, `<eos>`, `<pad>`, `<unk>`.

## Dataset: Flickr30k
- ~31k images, 5 captions each.
- Chosen over Flickr8k (too small/toy) and MS COCO (COCO's ~120k images would make
  each ablation training run much slower to iterate on; Flickr30k is the sweet spot
  for running multiple decoder/decoding-strategy ablations quickly). COCO is a
  possible later stretch goal once the pipeline and ablations are solid, not part
  of the initial build.

## Evaluation

### Standard metrics
BLEU-1–4, METEOR, ROUGE-L, CIDEr — computed via `pycocoevalcap`, so results are
directly comparable to published captioning papers.

### LLM-as-judge — DROPPED (2026-09-22)
Nihal decided to skip LLM-as-judge evaluation entirely. `src/judge.py`, the notebook
sections and every mention were removed. Evaluation is the standard metrics plus the
paired-bootstrap significance analysis in notebook 07.

## Interface / demo
**Built (2026-09-22): `app.py`.** Gradio app: upload an image, pick decoder
(Transformer / LSTM), training objective (cross-entropy / SCST) and decoding
(greedy / beam, defaulting to the val-tuned settings), and get the caption plus a
word-by-word attention map. Models load lazily from `checkpoints/`.

## Workflow
Building with Claude Code, working directly in the project directory. This is a
**personal project**, kept simple rather than over-engineered — not building it as
a polished production repo with configs/, CI, etc. It will be pushed to GitHub only
once it's in a reasonably finished state, so commits don't need to be as
incremental/step-by-step as the RAG project's; a normal, straightforward file
layout is preferred over the more elaborate `src/` module split.

**Notebooks over scripts.** Nihal prefers `.ipynb` notebooks over `.py` files for
the actual pipeline work (data prep, training, evaluation, ablations) — this is not
a from-scratch-modules project like the RAG one. Shared code that would otherwise
be duplicated across notebooks (model classes, the dataset/vocab loader) goes in a
small `src/` package that notebooks import from, just enough to avoid copy-pasting
the same class into six notebooks. The one likely exception is the Gradio demo
later — Gradio apps typically run as a script (`app.py`) rather than a notebook.

## Suggested repo structure
```
image-captioning-flickr30k/
├── data/
│   ├── raw/                          # Flickr30k images + captions
│   └── processed/                    # extracted ResNet-50 features, vocab, splits
├── src/
│   ├── dataset.py                    # dataset class, vocab builder
│   ├── models.py                     # TransformerCaptioner, LSTMAttentionCaptioner (shared image side)
│   ├── decoding.py                   # greedy + batched beam search (shared by all decoders)
│   ├── train.py                      # shared XE training loop, run caching, checkpoints
│   ├── scst.py                       # CIDEr-D reward + self-critical sequence training
│   ├── viz.py                        # notebook display: plot style, styled tables, caption panels
│   └── utils.py                      # feature extraction, fast + official (pycocoevalcap) metrics
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # download data, extract ResNet-50 features, build vocab
│   ├── 02_transformer_decoder.ipynb  # decoder + training loop
│   ├── 03_decoding.ipynb             # greedy first, then beam search
│   ├── 04_evaluation.ipynb           # pycocoevalcap metrics on the test split
│   ├── 05_lstm_attention_decoder.ipynb  # ablation comparison model
│   ├── 06_scst_finetuning.ipynb      # SCST (CIDEr reward) for both decoders
│   └── 07_ablation_results.ipynb     # decoder × decoding × training-objective writeup
├── checkpoints/                      # saved model weights
├── results/                          # metrics/eval outputs per config
├── app.py                            # Gradio demo
├── README.md
└── requirements.txt
```

## Build order
1. `01_data_preprocessing.ipynb`: download Flickr30k, extract ResNet-50 grid features, build vocab.
2. `02_transformer_decoder.ipynb`: Transformer decoder + training loop.
3. `03_decoding.ipynb`: greedy decoding first to get a caption out end-to-end; beam search after.
4. `04_evaluation.ipynb`: standard metrics, re-runnable across configs.
5. `05_lstm_attention_decoder.ipynb`: LSTM + Attention decoder as the comparison model.
6. `06_scst_finetuning.ipynb`: SCST fine-tuning of both XE models (added 2026-09-16, see below).
7. `07_ablation_results.ipynb`: decoder × decoding × training-objective writeup with bootstrap CIs.
8. Gradio demo (`app.py`, later).

Execution order differs from reading order: 02 → 05 → 06 → 03 → 04 → 07 (04 evaluates every
checkpoint that exists). **Run GPU notebooks one at a time**: each training kernel holds ~5 GB of
private RAM plus the feature memmap cache, and two at once pushed the 23 GB laptop into heavy swapping.

## Implementation decisions (made 2026-09-16, before notebook 01)
- **Environment:** existing global Python 3.13 (torch 2.13+cu130 already works on the
  RTX 5050); dependencies tracked in `requirements.txt`. Java (Temurin 21 JRE) installed
  for pycocoevalcap's METEOR + PTB tokenizer.
- **Data layout:** raw data in `data/raw/` (`Images/`, `captions.txt`,
  `dataset_flickr30k.json`). The Kaggle download's duplicate image folder and
  `results.csv` were deleted.
- **Split:** Karpathy split (29,000 train / 1,014 val / 1,000 test), with captions taken
  from `dataset_flickr30k.json` (complete 5 captions per image; the Kaggle
  `captions.txt` is missing one).
- **Features:** ResNet-50 7×7×2048 grid features stored as one float16 `.npy` memmap.
- **Vocab:** lowercase, punctuation stripped, min word frequency 5 (train split only).
- **Transformer decoder:** image grid → linear projection to d_model + learned 2D
  (row + column) positional embedding, no trainable encoder layers (keeps the LSTM
  ablation fair). 3 decoder layers, d_model 512, 8 heads, tied input/output embeddings.
- **Training:** AdamW + linear warmup + cosine decay, label smoothing 0.1, all 5
  captions per image per epoch, best checkpoint chosen by validation CIDEr (greedy).
- **Workflow:** Claude writes *and* executes the notebooks. No git until push time.
  First session goes through notebook 03, then stops for review.
- **Question style:** open decisions are asked all at once as multiple-choice
  questions, then work proceeds.

## Decisions made autonomously (2026-09-16, while Nihal was away; he asked to "maximize accuracy")
Review these; each is easy to revert.
- **Regularization sweep for both decoders:** the first Transformer run overfit (val loss
  minimum at epoch 7). Each decoder is trained with the same 4 settings (dropout 0.1/0.3/0.5,
  weight decay 0.01/0.1); the best val-CIDEr run becomes `checkpoints/{decoder}_xe.pt`.
  Architecture (layers, width, heads) was NOT changed.
- **Length-bucketed batches** (`LengthBucketSampler`): padded length 35 → 15 tokens, ~2× faster
  LSTM training. Used for every run (the first baseline was retrained with it for consistency).
- **LSTM + attention decoder:** Show-Attend-Tell style (additive attention, β gate, deep output,
  tied embeddings), hidden 1024 → 16.4M params vs Transformer 17.5M (parameter parity).
- **SCST extension (new notebook 06):** CIDEr-D reward with train-set document frequencies,
  5 samples/image, mean-of-other-samples baseline, `<eos>`-aware reward, Adam lr 2e-5, 15 epochs,
  same recipe for both decoders. Adds a third ablation axis (XE vs SCST).
- **Evaluation protocol (notebook 04):** beam size / length penalty tuned per model on val
  (fast CIDEr), test reported with official pycocoevalcap metrics ×100, plus human leave-one-out
  and published Flickr30k numbers for context.
- **Display:** all notebooks use `src/viz.py` (styled tables, JPEG caption panels, live
  per-epoch tables) instead of raw prints and oversized PNG figures.

## Status (2026-09-16, end of day)
All 7 notebooks are executed end to end, with written summaries; README.md has the results.
- **Val (greedy) CIDEr, XE:** Transformer 0.476 (dropout 0.3 + wd 0.1), LSTM 0.463 (dropout 0.3).
  **After SCST:** 0.559 and 0.548.
- **Test (official ×100), best:** Transformer + SCST + beam (k=7, α=0.5) with BLEU-4 25.3,
  METEOR 20.3, ROUGE-L 47.1, CIDEr 58.3.
- **Ablation (paired bootstrap):**
  - SCST +5.6–9.2 CIDEr, significant everywhere.
  - Beam +1.3–4.1, significant everywhere.
  - Transformer vs LSTM +1.1–4.0, significant only for XE + beam.
- **Engineering notes learned along the way:**
  - Greedy decoding only reproduces a saved val score at the training-time batch size (256),
    because bf16 kernels change with batch shape.
  - `TransformerCaptioner.step` projects only the last position (full-prefix logits for
    256×10 beams ran out of GPU memory).
  - `src/viz.table` must receive all column formats in one `Styler.format` call.
  - On Windows, write files with `encoding="utf-8"`.
- **2026-09-22 cleanup:** Gemini judge removed everywhere; notebook markdown cut to short
  section intros + a 4-bullet summary each; comparison figures show at most 2 models + one
  human caption (`viz.show_captions` row height now accounts for the row gap, which fixes the
  overlapping text); `app.py` demo added.
- **Demo:** `python app.py` (tested, Gradio 6: pass `theme=` to `launch()`, not `Blocks()`).
- **Not done yet:** GitHub push. A `.gitignore` excluding data/ and checkpoints/ is in place.

## Open / not yet decided
- Final decision on whether to attempt COCO as a stretch goal — deferred until after
  the Flickr30k pipeline and ablations are solid.
- Exact GitHub push timing — deferred until the project is in a reasonably finished
  state; no need for the RAG project's incremental-commit cadence along the way.
