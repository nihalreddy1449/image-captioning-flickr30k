# Image captioning on Flickr30k

Give the model a photo, get a sentence back. The interesting part is not that it works, but which of the usual design choices actually change the result. Three things normally get changed together in captioning projects, so this one changes them one at a time and measures each:

| axis | levels |
|---|---|
| decoder architecture | Transformer decoder vs LSTM with soft attention |
| decoding strategy | greedy vs beam search, with beam size and length penalty tuned on validation |
| training objective | cross-entropy vs cross-entropy followed by self-critical sequence training |

Both decoders are written from scratch in PyTorch, multi-head attention included, and they read the same frozen ResNet-50 features. Everything was trained on one 8 GB laptop GPU (RTX 5050).

## Demo

```bash
python app.py     # http://127.0.0.1:8000
```

Upload a photo or click one of the samples, then choose the decoder, the training objective and the decoding strategy. The page returns the caption and a strip of attention maps showing which part of the image the decoder was reading as it produced each word. Beam settings default to the ones tuned on validation.

## Results

Karpathy test split, 1,000 images, scored with `pycocoevalcap` and multiplied by 100 the way papers report it. Beam settings were tuned per model on validation, never on test.

| decoder | training | decoding | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|
| Transformer | cross-entropy | greedy | 65.4 | 22.8 | 19.6 | 45.4 | 48.7 |
| Transformer | cross-entropy | beam (k=7, α=1) | 65.4 | 23.9 | 20.3 | 46.2 | 52.7 |
| Transformer | + SCST | greedy | 67.8 | 24.4 | 20.1 | 46.7 | 56.6 |
| Transformer | + SCST | beam (k=7, α=0.5) | 68.4 | 25.3 | 20.3 | 47.1 | 58.3 |
| LSTM + attention | cross-entropy | greedy | 62.3 | 20.4 | 19.5 | 44.8 | 46.3 |
| LSTM + attention | cross-entropy | beam (k=5, α=1) | 62.3 | 21.0 | 20.3 | 45.5 | 48.8 |
| LSTM + attention | + SCST | greedy | 67.9 | 24.1 | 20.1 | 46.6 | 55.5 |
| LSTM + attention | + SCST | beam (k=2, α=0.5) | 68.1 | 24.7 | 20.2 | 46.7 | 56.8 |

Published Flickr30k numbers, as reported in those papers. Their encoders differ and two of them fine-tune the CNN, so read this as context rather than a head-to-head:

| system | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|
| Show, Attend and Tell, hard attention (VGG-19) | 19.9 | 18.5 | not reported |
| Adaptive attention (ResNet-152, fine-tuned) | 25.1 | 20.4 | 53.1 |
| this project, Transformer + SCST + beam | 25.3 | 20.3 | 58.3 |

## What each axis is worth

Every effect below is a difference in test CIDEr between two configurations that differ in one factor. The intervals come from a paired bootstrap over the 1,000 test images, so they say how much of the difference could be an accident of which images ended up in the test split.

| axis | effect on CIDEr | holds up? |
|---|---|---|
| SCST vs cross-entropy | +5.6 to +9.2 | yes, in all four settings |
| beam search vs greedy | +1.3 to +4.1 | yes, in all four settings |
| Transformer vs LSTM + attention | +1.1 to +4.0 | only for cross-entropy with beam search |

Three things stood out.

The training objective decides far more than the architecture does. Under cross-entropy the Transformer leads the LSTM by 2 to 4 CIDEr, which is the comparison most projects stop at. After SCST the gap falls to about 1.5 and the bootstrap interval covers zero, so on this dataset the two decoders end up equivalent once both are trained against the metric.

Beam search pays off less after SCST (+1.3 to +1.8, against +2.5 to +4.1 under cross-entropy), which fits: SCST has already pushed probability mass toward the captions CIDEr likes, so searching harder finds less. The length penalty matters more than the beam width. At α=0 the model scores raw log-probability, wider beams return shorter captions, and CIDEr drops from 0.491 at k=3 to 0.452 at k=10.

Regularization was worth as much as some architecture decisions. Both baselines overfit within 8 epochs. Raising dropout to 0.3 and weight decay to 0.1 added 7.5% validation CIDEr for the Transformer and 4.8% for the LSTM, before any of the ablation work started.

SCST has a cost that the metrics hide. Captions drift toward phrasing the reward likes, and the vocabulary across the test set shrinks from 578 distinct words to 400. The reward includes an explicit end token, which keeps captions from trailing off mid-phrase (0% of them do).

## How it works

The encoder is a frozen ImageNet ResNet-50. Each image is squashed to 224×224 (no center crop, which would cut off things the captions mention) and the last conv block gives a 7×7 grid of 2048-d vectors. All 31,014 images in the split are encoded once, in about 1.4 minutes, and stored as a 6.2 GB float16 memmap, so no notebook after the first one touches a JPEG.

Both decoders see that grid through the same layer: `Linear(2048→512)`, `LayerNorm`, plus a learned 2D position built from a row and a column embedding. Only the decoder differs, which is what makes the comparison fair.

The Transformer decoder (17.5M parameters) has 3 pre-LayerNorm blocks: masked self-attention, cross-attention over the 49 grid cells, then a GELU feed-forward layer, with the output projection tied to the word embedding. Multi-head attention is written by hand. Training goes through PyTorch's fused `scaled_dot_product_attention`, and a second explicit path returns the attention weights for the visualizations.

The LSTM decoder (16.4M parameters, sized to match) follows Show, Attend and Tell. At each word it scores the grid with additive attention against the previous hidden state, gates the resulting context vector, runs an `LSTMCell` over the word and context, and produces logits through a deep output layer with tied embeddings.

Cross-entropy training uses AdamW with one warmup epoch and cosine decay, label smoothing 0.1, and length-bucketed batches that cut the padded batch length from 35 tokens to 15. Checkpoints are chosen by validation CIDEr rather than validation loss, which matters here because in the unregularized runs loss starts rising around epoch 8 while CIDEr keeps improving. Each decoder gets the same four regularization settings and its best run is the one that goes forward.

SCST then samples 5 captions per image, scores each with CIDEr-D, and raises the probability of the ones that beat the mean of the other four. Document frequencies come from the training set, and the reward implementation matches `pycocoevalcap` to six decimals. An explicit end token in the reward prevents the classic failure where captions stop mid-phrase.

Decoding is batched for both models: greedy and beam search with a length penalty, sharing one `init_state` / `step` interface so the same search code drives either decoder.

For evaluation the notebooks use the official `pycocoevalcap` metrics (PTB tokenizer, BLEU, METEOR, ROUGE-L, CIDEr-D) and a paired bootstrap for every comparison. One human reference scored against the other four reaches CIDEr 68.2 and BLEU-4 19.0, which is a useful reminder that beating humans on n-gram overlap says less than it sounds.

## Notebooks

| notebook | what it does |
|---|---|
| `01_data_preprocessing` | Karpathy split, caption cross-checks, 7,414-word vocabulary, ResNet-50 features and their sanity checks |
| `02_transformer_decoder` | the model, pre-training checks (loss at init, causal-mask leakage, overfitting one batch), regularization sweep |
| `03_decoding` | greedy vs beam search: the sweep, caption statistics, where beam helps and hurts, attention maps |
| `04_evaluation` | per-model decoding tuning on validation, official test metrics, human and published baselines |
| `05_lstm_attention_decoder` | the comparison decoder with the same checks and the same sweep |
| `06_scst_finetuning` | self-critical fine-tuning for both decoders |
| `07_ablation_results` | effect sizes with bootstrap intervals, tuning budget, cost, caption statistics, examples |

Shared code sits in `src/`: `dataset.py`, `models.py`, `decoding.py`, `train.py`, `scst.py`, `utils.py` for metrics and feature extraction, and `viz.py` for the notebook tables and figures.

## Running it

1. Data. Put the Flickr30k images in `data/raw/Images/` and Karpathy's `dataset_flickr30k.json` in `data/raw/`. The JSON comes from [caption_datasets.zip](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
2. Environment. Install PyTorch with CUDA from pytorch.org, then `pip install -r requirements.txt`. METEOR and the PTB tokenizer shell out to Java, so a JRE 11+ needs to be on PATH.
3. Notebooks. Run them as `01 → 02 → 05 → 06 → 03 → 04 → 07`. Finished training runs are cached by checkpoint and history file, so re-running a notebook retrains only what is missing. Run one GPU notebook at a time: each kernel holds around 5 GB of RAM plus the feature cache.
4. Demo. `python app.py`.

```
data/raw/          Flickr30k images and captions            (not in git)
data/processed/    features, captions.json, vocab.json      (not in git)
checkpoints/       the four models plus every sweep run     (not in git)
results/           run histories, sweeps, metrics, figures
notebooks/         01 to 07
src/               shared code
web/               front-end for the demo
app.py             demo server
```

## Known limits

The encoder is frozen at 224×224, so the 7×7 grid is coarse and small objects get lost. Each configuration was trained with one seed, and the bootstrap covers test-set sampling rather than seed-to-seed variance. The Transformer decoder has no KV cache, so it re-runs the whole prefix at each step and decodes about four times slower than the LSTM (927 vs 3,703 images per second, greedy, on the test split).
