"""Notebook presentation helpers: one plot style, styled tables, caption panels, attention maps.

Photo-heavy figures are sent to the notebook as JPEG (a few hundred KB instead of several MB of PNG).
"""
import io
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from IPython.display import HTML, Image as IPyImage, display
from matplotlib.transforms import offset_copy
from PIL import Image

# Categorical slots, always assigned in this order (validated colorblind-safe palette).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e4e3df"

HIGHLIGHT = "background-color: #d9e8fb; color: #0b0b0b; font-weight: 700"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "axes.prop_cycle": cycler(color=SERIES),
    })
    pd.set_option("display.max_colwidth", 120)


def table(df, precision=3, maximize=(), minimize=(), caption=None, index=False, percent=(), formats=None):
    """Styled DataFrame: text left / numbers right, fixed precision, best value per listed column highlighted."""
    # One formatter per column in a single call: pandas resets columns missing from a later dict format.
    fmt = {}
    for c in df.columns:
        if formats and c in formats:
            fmt[c] = formats[c]
        elif c in percent:
            fmt[c] = "{:.1%}"
        elif pd.api.types.is_float_dtype(df[c]):
            fmt[c] = f"{{:,.{precision}f}}"
        elif pd.api.types.is_integer_dtype(df[c]):
            fmt[c] = "{:,}"
    sty = df.style.format(fmt, escape="html", na_rep="\u2013")
    sty = sty.format_index(escape="html", axis=1)
    for col in maximize:
        sty = sty.highlight_max(subset=[col], props=HIGHLIGHT)
    for col in minimize:
        sty = sty.highlight_min(subset=[col], props=HIGHLIGHT)
    if not index:
        sty = sty.hide(axis="index")
    styles = [
        {"selector": "th", "props": "padding: 4px 12px; border-bottom: 1px solid #c3c2b7"},
        {"selector": "td", "props": "padding: 4px 12px; font-variant-numeric: tabular-nums"},
    ]
    if caption:
        sty = sty.set_caption(caption)
        styles.append({"selector": "caption",
                       "props": "caption-side: top; text-align: left; font-weight: 700; font-size: 1.05em; padding: 0 0 6px 0"})
    sty = sty.set_table_styles(styles)
    # Per-column alignment (column-class selectors outrank the generic td rule): text left, numbers right.
    align = {c: [{"selector": "", "props": "text-align: " + (
        "right" if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) else "left")}]
        for c in df.columns}
    return sty.set_table_styles(align, overwrite=False)


def show_jpeg(fig, quality=90):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", pil_kwargs={"quality": quality})
    plt.close(fig)
    display(IPyImage(data=buf.getvalue(), format="jpeg"))


def _thumbnail(path, size):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    return img


def show_captions(examples, image_dir, ncols=2, wrap=46, thumb=420, colors=None, title=None):
    """Grid of image + caption panels.

    examples: [{"filename": ..., "title": optional header, "captions": [(label, text), ...]}]
    colors: {label: color} for the small label above each caption (defaults to series order).
    """
    labels = list(dict.fromkeys(label for ex in examples for label, _ in ex["captions"]))
    colors = {**{lab: SERIES[i % len(SERIES)] for i, lab in enumerate(labels)}, **(colors or {})}
    line_pts, hspace = 13.5, 0.22
    n_lines = max(
        (1.6 if ex.get("title") else 0)
        + sum(1.1 + len(textwrap.wrap(text, wrap) or [""]) + 0.45 for _, text in ex["captions"])
        for ex in examples
    )
    # The text block is drawn in points from the top of its axes, so the row must be tall enough to
    # hold every line plus the gap between rows - otherwise long captions spill into the row below.
    row_h = max(2.6, (n_lines * line_pts / 72) * (1 + hspace) + 0.25)
    nrows = math.ceil(len(examples) / ncols)

    fig = plt.figure(figsize=(8.4 * ncols, row_h * nrows + (0.4 if title else 0)))
    gs = fig.add_gridspec(nrows, 2 * ncols, width_ratios=[1.0, 1.3] * ncols, wspace=0.05, hspace=hspace)
    for k, ex in enumerate(examples):
        r, c = divmod(k, ncols)
        ax_img = fig.add_subplot(gs[r, 2 * c])
        ax_img.imshow(_thumbnail(Path(image_dir) / ex["filename"], thumb))
        ax_img.set_anchor("NE")
        ax_img.axis("off")

        ax_txt = fig.add_subplot(gs[r, 2 * c + 1])
        ax_txt.axis("off")
        line = 0.0

        def put(text, **kw):
            tr = offset_copy(ax_txt.transAxes, fig=fig, x=8, y=-line * line_pts, units="points")
            ax_txt.text(0, 1, text, transform=tr, va="top", ha="left", **kw)

        if ex.get("title"):
            put(ex["title"], size=10.5, weight="bold", color=INK)
            line += 1.6
        for label, text in ex["captions"]:
            put(label.upper(), size=8, weight="bold", color=colors[label])
            line += 1.1
            for chunk in textwrap.wrap(text, wrap) or [""]:
                put(chunk, size=10, color=INK)
                line += 1
            line += 0.45
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=13, weight="bold")
    show_jpeg(fig)


def show_attention(image_path, words, attn, grid=7, cols=7, title=None, size=224):
    """Where the decoder looks for each word. attn: [len(words), grid*grid] weights.

    Rendered as a spotlight: the image stays bright where attention is high and fades elsewhere.
    """
    img = np.asarray(Image.open(image_path).convert("RGB").resize((size, size))).astype(np.float32) / 255
    rows = math.ceil((len(words) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.75 * cols, 2.15 * rows + (0.45 if title else 0)),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.06})
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(img)
    axes[0].set_title("input", fontsize=9.5, color=INK_2, loc="center", weight="normal")
    for ax, word, a in zip(axes[1:], words, attn):
        a = np.asarray(a, dtype=np.float32).reshape(grid, grid)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        heat = Image.fromarray(np.uint8(255 * a)).resize((size, size), Image.BICUBIC)
        mask = np.asarray(heat, dtype=np.float32)[..., None] / 255
        ax.imshow(img * (0.15 + 0.85 * mask))
        ax.set_title(word, fontsize=10.5, loc="center", weight="bold")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, x=0.125, y=1.0, ha="left", fontsize=11.5, weight="bold")
    show_jpeg(fig)


def note(markdown_text):
    """A short highlighted result line under a cell (renders in VS Code, Jupyter and GitHub)."""
    display(HTML(f"<div style='padding:6px 10px;border-left:3px solid {SERIES[0]};margin:4px 0'>{markdown_text}</div>"))


class LiveTable:
    """Per-epoch training log that updates in place (the saved notebook keeps the final table)."""

    def __init__(self, caption, columns, maximize=(), minimize=(), formats=None):
        self.caption, self.columns, self.maximize, self.minimize = caption, columns, maximize, minimize
        self.formats = {columns[k]: v for k, v in (formats or {}).items()}
        self.handle = display(HTML(f"<b>{caption}</b>: starting…"), display_id=True)

    def __call__(self, history):
        df = pd.DataFrame(history)[list(self.columns)].rename(columns=self.columns)
        maxi = [self.columns[c] for c in self.maximize]
        mini = [self.columns[c] for c in self.minimize]
        self.handle.update(table(df, precision=3, maximize=maxi, minimize=mini, caption=self.caption, formats=self.formats))
