import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRAIN_FILE_PATH = "train.csv" 
BASE_FEAT_COLS = [
    "canonical_entropy",
    "canonical_margin",
    "canonical_z_logp",
    "canonical_selected_rank",
    "canonical_logprobs",
    "canonical_logit_gap",
    "canonical_topk_mass@5",
    "canonical_topk_mass@10",
    "canonical_d_entropy",
    "canonical_d_margin",
    "canonical_d_logp",
]

# Columns treated as rank-like (log1p before z-score)
RANK_LIKE_COLS = {"canonical_selected_rank"}


def safe_load_list(cell):
    """Parse a JSON list or return the list itself; otherwise None."""
    if isinstance(cell, list):
        return cell
    try:
        val = json.loads(cell)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    return None


def extract_base_arrays(row: pd.Series, feat_cols: List[str]) -> Dict[str, np.ndarray]:
    """Extract aligned feature arrays from a row, truncating to the shortest length."""
    def arr(name):
        v = safe_load_list(row.get(name))
        if v is None:
            return np.asarray([], dtype=np.float32)
        return np.asarray([float(x) for x in v], dtype=np.float32)

    base = {c: arr(c) for c in feat_cols}
    lens = [len(v) for v in base.values() if len(v) > 0]
    T = min(lens) if lens else 0
    if T <= 0:
        return {k: np.asarray([], dtype=np.float32) for k in base}
    return {k: v[:T] for k, v in base.items()}


def zscore_fit(df: pd.DataFrame, feat_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean/std for each feature column over all time steps (with rank-like log1p).
    Returns {column: (mean, std)} using std=1.0 when variance is degenerate.
    """
    stats = {c: (0.0, 1.0) for c in feat_cols}
    accum = {c: [] for c in feat_cols}

    for _, row in df.iterrows():
        seqs = extract_base_arrays(row, feat_cols)
        lens = [len(v) for v in seqs.values() if len(v) > 0]
        if not lens:
            continue
        T = min(lens)
        for c in feat_cols:
            a = seqs.get(c, np.asarray([], dtype=np.float32))
            if a.size == 0:
                continue
            a = a[:T].astype(np.float32)
            if c in RANK_LIKE_COLS:
                a = np.log1p(np.maximum(0.0, a))
            accum[c].extend(a.tolist())

    for c in feat_cols:
        vals = accum[c]
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std <= 1e-12:
            std = 1.0
        stats[c] = (mean, std)

    return stats
