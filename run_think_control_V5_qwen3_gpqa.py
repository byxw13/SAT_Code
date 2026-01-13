# run_with_R_detector_v2.py ? sentence-end steps, branching, and dual-anchor rollback
# -*- coding: utf-8 -*-
"""
Online R detector runner with detailed branching logs, compatible with the extended + derived feature R model.
- A "step" is one sentence end (default .?!; or newline).
- At each sentence end, score R and maintain two anchors:
    * best_anchor: sentence end with the maximum historical R
    * latest_anchor: most recent sentence end with R ? TAU_ANCHOR
  If the current R drops below the active_anchor (=latest or best) by more than DELTA_DROP, rollback to that anchor and rewrite the whole sentence (baseline token included).
- Main chain and branches both use repetition penalty; pack?token alignment is exact.
- Canonical and temporal features are computed online (window/threshold from meta or defaults).
- If latest_anchor is not refreshed after rollback, count it; after two misses, disable rollback (likely out-of-scope question).
- Logging:
  1) keynode_action: candidate vs. R before/after per sentence + anchor state
  2) branch_decision: baseline vs. chosen full sentence (token ids and readable delta)
- After finishing, compute full reasoning-chain R and judge correctness by threshold (meta.best_tau or 0.9).
"""

import os, json, time, math, argparse, re, sys
from typing import List, Dict, Any, Tuple, Optional
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models

from tool import (
    TRAIN_FILE_PATH,
    BASE_FEAT_COLS,
    RANK_LIKE_COLS,
    zscore_fit,
)

# =====================
# =====================
K_TOP = 512
RANK_TRUNC_SENTINEL = 513
ZWIN                  = 50   # ### NEW  z-score 
TOPK_LIST             = [5, 10]  # ### NEW top-k 

STEP_TOKEN = "\n"                   #  PRM  step_token 
SENT_END_REGEX = re.compile(r"$^")  # “”，
COUNT_NEWLINE_AS_END = False        #  STEP_TOKEN ，
MAX_SENT_TOKENS = 10**9             # “”

TEMPERATURE = 0.6  # Changed to greedy decoding for main chain
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 16384  # Aligned with baseline code

BRANCH_TOPB = 5
BRANCH_TEMP = 0.6
BRANCH_SENTENCE = 1


GAMMA_DROP = 100
GAMMA_JUMP = 100

TAU_ANCHOR = 0.4     #  latest_anchor（“”）
DELTA_DROP = 0.22     # ：R_t <= anchor_R - DELTA_DROP → 
ROLLBACK_COOLDOWN = 6 #  1 
MAX_NO_REFRESH = 2    #  2  latest_anchor  → 
MAX_REFRESH = 3


INPUT_DATA_FILE = "gpqa_test.jsonl"
OUT_JSONL = "r_gpqa_Qwen3_4B_seed3407_sat.jsonl"

ONLINE_ZWIN_FOR_LOGP = 50
GTE_SMALL_PATH = "Gte-Small"
ZSTATS_PATH = "zstats.json"

# =====================
# =====================
MAX_THINK_ENTERS = 2
THETA_ENTER = 0.3
K_ENTER = 3

THETA_GOOD = 0.8
THETA_BAD_1 = 0.15
THETA_BAD_2 = 0.40

FAST_STEP_R_THRESHOLD = 0.95   # R  → ， FAST
SLOW_STEP_R_THRESHOLD = 0.3   # R  → ， SLOW

K_GOOD = 10
K_BAD = 6
K_FLAT = 10
K_DOWN = 8

DELTA_DOWN_MIN = 0.03
DELTA_FLAT = 0.05

THINK_SUPPRESS_STRENGTH = 1e4

MIN_BASE_SENT_STAY  = 15   # BASE ， THINK
MIN_THINK_SENT_STAY = 20   # THINK ， BASE

SEED = 3407
def print_cuda_mem(tag: str = ""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA N/A"); return
    d = torch.cuda.current_device(); torch.cuda.synchronize()
    a = torch.cuda.memory_allocated(d); r = torch.cuda.memory_reserved(d); p = torch.cuda.max_memory_allocated(d)
    to_mb = lambda x: f"{x/1024/1024:.1f} MB"
    print(f"，[{tag}] alloc={to_mb(a)}, reserved={to_mb(r)}, peak={to_mb(p)}")

def _viz_text(s: str, maxlen: int = 400) -> str:
    """，/。"""
    s = s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    s = s.replace(" ", "·")
    if len(s) > maxlen:
        s = s[:maxlen] + " …"
    return s

# =====================
# =====================
def setup_seed_global(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# =====================
# =====================
def step_features_from_rawlogits(raw_bucket: List[torch.Tensor],
                                 gen_ids: List[int],
                                 topk_list: List[int] = TOPK_LIST,
                                 zwin: int = ZWIN) -> Dict[str, List[float]]:
    T = len(gen_ids)
    assert len(raw_bucket) >= T, f"bucket steps {len(raw_bucket)} < gen tokens {T}"

    sel_logp, sel_rank = [], []
    margin_list, gap_list, entropy_list = [], [], []
    topk_masses = {k: [] for k in topk_list}

    fallback_count = 0

    for i in range(T):
        pack = raw_bucket[i]                # dict:  "topk_vals" / "topk_idx"
        topk_vals = pack["topk_vals"]       # [1,K] CPU（）
        topk_idx  = pack["topk_idx"]        # [1,K] CPU
        tid = int(gen_ids[i])

        local_logp = F.log_softmax(topk_vals.float(), dim=-1)  # [1,K]
        local_p    = torch.exp(local_logp)                     # [1,K]

        in_topk_mask = (topk_idx[0] == tid)
        assert bool(in_topk_mask.any()), (
            f"[E] step {i}: token id {tid} not in top-K. "
            f"Increase K or check sampling/logits path."
        )

        pos = int(in_topk_mask.nonzero(as_tuple=False)[0].item())
        sel_logp.append(float(local_logp[0, pos].item()))
        sel_rank.append(pos + 1)

        gap_list.append(
            float(topk_vals[0, 0].item() - topk_vals[0, 1].item())
            if topk_vals.shape[1] > 1 else float("inf")
        )

        p1 = float(local_p[0, 0].item())
        p2 = float(local_p[0, 1].item()) if local_p.shape[1] > 1 else 0.0
        margin_list.append(p1 - p2)

        entropy_list.append(float(-(local_p * local_logp).sum().item()))

        for k in topk_list:
            kk = min(k, local_p.shape[1])
            topk_masses[k].append(float(local_p[0, :kk].sum().item()))

    def _diff(x):
        if len(x) == 0: return []
        out = [0.0]
        for t in range(1, len(x)):
            out.append(float(x[t] - x[t-1]))
        return out

    def _zscore(x, win: int):
        if len(x) == 0: return []
        out = []
        for t in range(len(x)):
            L = max(0, t - win + 1)
            seg = x[L:t+1]
            m = float(np.mean(seg))
            s = float(np.std(seg)) + 1e-8
            out.append(float((x[t] - m) / s))
        return out

    feats = {
        "canonical_logprobs": sel_logp,
        "selected_rank": sel_rank,
        "margin": margin_list,
        "logit_gap": gap_list,
        "entropy": entropy_list,
        "d_entropy": _diff(entropy_list),
        "d_margin": _diff(margin_list),
        "d_canonical_logp": _diff(sel_logp),
        "z_canonical_logp": _zscore(sel_logp, zwin),

        "stat_fallback_count": fallback_count,   #  0
        "stat_total_steps": T,
        "_rank_trunc_sentinel": RANK_TRUNC_SENTINEL,
    }

    for k, vals in topk_masses.items():
        feats[f"topk_mass@{k}"] = vals

    return feats

def base_feats_from_bucket_like_training(
    bucket: List[Dict[str, torch.Tensor]],
    ids: List[int],
    zwin_for_logp: int,
) -> Dict[str, List[float]]:
    """
    “”， bucket+ids  11  canonical_* 。
    """
    T = len(ids)
    assert len(bucket) >= T, f"packs {len(bucket)} < tokens {T}"

    raw_bucket = []
    for i in range(T):
        pack = bucket[i]
        raw_bucket.append({
            "topk_vals": pack["topk_vals"],  # [1,K]
            "topk_idx":  pack["topk_idx"],   # [1,K]
        })

    feats = step_features_from_rawlogits(
        raw_bucket,
        gen_ids=ids,
        topk_list=[5, 10],         # 
        zwin=zwin_for_logp,
    )

    base_feats = {
        "canonical_logprobs":      feats["canonical_logprobs"],
        "canonical_selected_rank": feats["selected_rank"],
        "canonical_margin":        feats["margin"],
        "canonical_logit_gap":     feats["logit_gap"],
        "canonical_entropy":       feats["entropy"],
        "canonical_topk_mass@5":   feats["topk_mass@5"],
        "canonical_topk_mass@10":  feats["topk_mass@10"],
        "canonical_d_entropy":     feats["d_entropy"],
        "canonical_d_margin":      feats["d_margin"],
        "canonical_d_logp":        feats["d_canonical_logp"],
        "canonical_z_logp":        feats["z_canonical_logp"],
    }
    return base_feats


def init_canonical_feat_buffer() -> Dict[str, List[float]]:
    return {c: [] for c in BASE_FEAT_COLS}


def truncate_canonical_feats(canonical_feats: Optional[Dict[str, List[float]]], target_len: int):
    if not canonical_feats:
        return
    for k, seq in canonical_feats.items():
        if isinstance(seq, list):
            canonical_feats[k] = seq[:target_len]


def recompute_canonical_feats_from_bucket(
    bucket: List[Dict[str, torch.Tensor]],
    ids: List[int],
    zwin_for_logp: int,
) -> Dict[str, List[float]]:
    base = base_feats_from_bucket_like_training(bucket, ids, zwin_for_logp)
    feats = init_canonical_feat_buffer()
    for k in feats.keys():
        if k in base:
            feats[k] = list(base[k])
    return feats


def update_canonical_feats_incremental(
    canonical_feats: Optional[Dict[str, List[float]]],
    pack: Optional[Dict[str, torch.Tensor]],
    tok_id: int,
    zwin_for_logp: int,
):
    if canonical_feats is None or pack is None:
        return

    topk_vals = pack.get("topk_vals")
    topk_idx = pack.get("topk_idx")
    if topk_vals is None or topk_idx is None:
        return

    topk_vals = topk_vals.float()
    topk_idx = topk_idx.long()

    local_logp = F.log_softmax(topk_vals, dim=-1)
    local_p = torch.exp(local_logp)

    tid = int(tok_id)
    in_topk_mask = (topk_idx[0] == tid)
    if not bool(in_topk_mask.any()):
        print(f"topk，id{tid}")
        for key in [
            "canonical_logprobs",
            "canonical_selected_rank",
            "canonical_logit_gap",
            "canonical_margin",
            "canonical_entropy",
            "canonical_topk_mass@5",
            "canonical_topk_mass@10",
        ]:
            seq = canonical_feats.setdefault(key, [])
            if seq:
                seq.append(seq[-1])
            else:
                seq.append(0.0)

        entropy_seq = canonical_feats.get("canonical_entropy", [])
        if len(entropy_seq) >= 2:
            d_entropy = entropy_seq[-1] - entropy_seq[-2]
        else:
            d_entropy = 0.0
        canonical_feats.setdefault("canonical_d_entropy", []).append(float(d_entropy))

        margin_seq = canonical_feats.get("canonical_margin", [])
        if len(margin_seq) >= 2:
            d_margin = margin_seq[-1] - margin_seq[-2]
        else:
            d_margin = 0.0
        canonical_feats.setdefault("canonical_d_margin", []).append(float(d_margin))

        logp_seq = canonical_feats.get("canonical_logprobs", [])
        if len(logp_seq) >= 2:
            d_logp = logp_seq[-1] - logp_seq[-2]
        else:
            d_logp = 0.0
        canonical_feats.setdefault("canonical_d_logp", []).append(float(d_logp))

        if logp_seq:
            L = len(logp_seq)
            L0 = max(0, L - zwin_for_logp)
            window = logp_seq[L0:L]
            m = float(np.mean(window))
            s = float(np.std(window)) + 1e-8
            canonical_feats.setdefault("canonical_z_logp", []).append(float((logp_seq[-1] - m) / s))
        return  #  token 

    pos = int(in_topk_mask.nonzero(as_tuple=False)[0].item())

    sel_logp = float(local_logp[0, pos].item())
    sel_rank = pos + 1
    canonical_feats.setdefault("canonical_logprobs", []).append(sel_logp)
    canonical_feats.setdefault("canonical_selected_rank", []).append(sel_rank)

    if topk_vals.shape[1] > 1:
        gap = float(topk_vals[0, 0].item() - topk_vals[0, 1].item())
    else:
        gap = float("inf")
    canonical_feats.setdefault("canonical_logit_gap", []).append(gap)

    p1 = float(local_p[0, 0].item())
    p2 = float(local_p[0, 1].item()) if local_p.shape[1] > 1 else 0.0
    margin = p1 - p2
    canonical_feats.setdefault("canonical_margin", []).append(margin)

    entropy = float(-(local_p * local_logp).sum().item())
    canonical_feats.setdefault("canonical_entropy", []).append(entropy)

    for k in (5, 10):
        kk = min(k, local_p.shape[1])
        mass = float(local_p[0, :kk].sum().item())
        canonical_feats.setdefault(f"canonical_topk_mass@{k}", []).append(mass)

    entropy_seq = canonical_feats.get("canonical_entropy", [])
    if len(entropy_seq) >= 2:
        d_entropy = entropy_seq[-1] - entropy_seq[-2]
    else:
        d_entropy = 0.0
    canonical_feats.setdefault("canonical_d_entropy", []).append(float(d_entropy))

    margin_seq = canonical_feats.get("canonical_margin", [])
    if len(margin_seq) >= 2:
        d_margin = margin_seq[-1] - margin_seq[-2]
    else:
        d_margin = 0.0
    canonical_feats.setdefault("canonical_d_margin", []).append(float(d_margin))

    logp_seq = canonical_feats.get("canonical_logprobs", [])
    if len(logp_seq) >= 2:
        d_logp = logp_seq[-1] - logp_seq[-2]
    else:
        d_logp = 0.0
    canonical_feats.setdefault("canonical_d_logp", []).append(float(d_logp))

    if logp_seq:
        L = len(logp_seq)
        L0 = max(0, L - zwin_for_logp)
        window = logp_seq[L0:L]
        m = float(np.mean(window))
        s = float(np.std(window)) + 1e-8
        canonical_feats.setdefault("canonical_z_logp", []).append(float((logp_seq[-1] - m) / s))


def build_sentence_spans_and_texts(
    sent_history: List[Dict[str, Any]],
    extra_entries: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    spans: List[Tuple[int, int]] = []
    texts: List[str] = []
    entries: List[Dict[str, Any]] = list(sent_history)
    if extra_entries:
        entries.extend(extra_entries)

    for entry in entries:
        tokens = entry.get("tokens") or []
        if not tokens:
            continue
        start = entry.get("start_gen_len")
        if start is None:
            continue
        start = int(start)
        end = start + len(tokens) - 1
        if end < start:
            continue
        spans.append((start, end))
        texts.append(entry.get("text", ""))

    return spans, texts



class StepSeqPRM_GRU(nn.Module):
    def __init__(self, d_logits, d_text, d_model=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_logits + d_text, d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.out_head = nn.Linear(d_model, 1)  #  step  logit

    def forward(self, x_logits, x_text, lengths):
        """
        x_logits: [B, T, D_l]
        x_text:   [B, T, D_e]
        lengths:  [B] （ pad）
        """
        x = torch.cat([x_logits, x_text], dim=-1)  # [B, T, D_l+D_e]
        x = self.input_proj(x)                     # [B, T, d_model]

        lengths_cpu = lengths.cpu()
        sorted_len, sort_idx = torch.sort(lengths_cpu, descending=True)
        x_sorted = x[sort_idx]

        packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_len, batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.gru(packed)
        out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # [B, T_max, d_model]（T_max=）

        _, inv_idx = torch.sort(sort_idx)
        out = out_sorted[inv_idx]  # [B, T_max, d_model]

        out = self.dropout(out)
        logits = self.out_head(out).squeeze(-1)  # [B, T_max]
        return logits


def encode_step_texts(step_texts: List[str], encoder: SentenceTransformer) -> np.ndarray:
    """
     gte-small  step  embedding。
    """
    embs = encoder.encode(
        step_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embs.astype(np.float32)  # [S, D_e]

def filter_for_R(gen_ids: List[int],
                 bucket: List[Dict[str, torch.Tensor]],
                 ignore_ids: set[int]
                 ) -> tuple[list[int], list[Dict[str, torch.Tensor]]]:
    """ gen_ids  bucket  ignore_ids  token """
    assert len(bucket) >= len(gen_ids), "bucket  gen_ids "

    keep_idx = [i for i, t in enumerate(gen_ids) if t not in ignore_ids]
    if not keep_idx:
        return [], []

    filtered_ids = [gen_ids[i] for i in keep_idx]
    filtered_bucket = [bucket[i] for i in keep_idx]
    return filtered_ids, filtered_bucket



def r_prob_for_prefix(
    step_model: nn.Module,
    encoder: SentenceTransformer,
    tokenizer,
    prompt_ids: List[int],
    bucket: List[Dict[str, torch.Tensor]],
    gen_ids: List[int],
    feat_cols: List[str],
    zstats: Dict[str, Tuple[float, float]],
    rank_like_cols: set,
    device: torch.device,
    zwin_for_logp: int = ONLINE_ZWIN_FOR_LOGP,
    step_cache: Optional[Dict[str, Any]] = None,  # ：step embedding
    canonical_feats: Optional[Dict[str, List[float]]] = None,
    sentence_spans: Optional[List[Tuple[int, int]]] = None,
    step_texts_external: Optional[List[str]] = None,
) -> float:
    """
     2.0  GRU + gte-small ""：
    -  + （）（/）
    """
    if not gen_ids:
        return 0.5

    T_tokens = len(gen_ids)
    use_incremental = (
        canonical_feats is not None
        and sentence_spans is not None
        and step_texts_external is not None
    ) 

    used_feat_cols: List[str] = []
    spans_local: List[Tuple[int, int]] = []
    step_texts_local: List[str] = []
    T = 0
    mat: Optional[np.ndarray] = None

    if use_incremental:
        used_feat_cols = [
            c for c in feat_cols
            if c in canonical_feats and isinstance(canonical_feats[c], list) and len(canonical_feats[c]) > 0
        ]
        if not used_feat_cols:
            return 0.5
        T = min(min(len(canonical_feats[c]) for c in used_feat_cols), T_tokens)
        if T <= 0:
            return 0.5
        Fdim = len(used_feat_cols)
        mat = np.zeros((T, Fdim), dtype=np.float32)
        for j, c in enumerate(used_feat_cols):
            arr = np.asarray(canonical_feats[c][:T], dtype=np.float32)
            if c in rank_like_cols:
                arr = np.log1p(np.maximum(0.0, arr))
            m, s = zstats.get(c, (0.0, 1.0))
            s = 1.0 if s <= 1e-12 else s
            mat[:, j] = (arr - m) / s

        filtered_spans: List[Tuple[int, int]] = []
        filtered_texts: List[str] = []
        for idx, span in enumerate(sentence_spans):
            if span is None or len(span) != 2:
                continue
            st, ed = int(span[0]), int(span[1])
            if st >= T:
                continue
            ed = min(ed, T - 1)
            if st > ed:
                continue
            filtered_spans.append((st, ed))
            filtered_texts.append(step_texts_external[idx] if idx < len(step_texts_external) else "")
        if not filtered_spans:
            filtered_spans = [(0, T - 1)]
            fallback_text = step_texts_external[0] if step_texts_external else ""
            filtered_texts = [fallback_text]
        spans_local = filtered_spans
        step_texts_local = filtered_texts
    else:
        base_feats = base_feats_from_bucket_like_training(
            bucket=bucket,
            ids=gen_ids,
            zwin_for_logp=zwin_for_logp,
        )
        used_feat_cols = [
            c for c in feat_cols
            if c in base_feats and isinstance(base_feats[c], list) and len(base_feats[c]) > 0
        ]
        if not used_feat_cols:
            return 0.5
        T = min(min(len(base_feats[c]) for c in used_feat_cols), T_tokens)
        if T <= 0:
            return 0.5
        Fdim = len(used_feat_cols)
        mat = np.zeros((T, Fdim), dtype=np.float32)
        for j, c in enumerate(used_feat_cols):
            arr = np.asarray(base_feats[c][:T], dtype=np.float32)
            if c in rank_like_cols:
                arr = np.log1p(np.maximum(0.0, arr))
            m, s = zstats.get(c, (0.0, 1.0))
            s = 1.0 if s <= 1e-12 else s
            mat[:, j] = (arr - m) / s

        sentence_spans_decoded: List[Tuple[int, int]] = []
        ctx_ids = list(prompt_ids)
        curr_start = 0
        curr_has_content = False

        for i, tok_id in enumerate(gen_ids[:T]):
            piece = decode_piece_in_context(tokenizer, ctx_ids, int(tok_id))
            ctx_ids.append(int(tok_id))

            payload = piece.replace("\r\n", "\n").replace("\n", "")
            if payload.strip() != "":
                curr_has_content = True

            end_by_step = piece_has_sentence_end(piece) and curr_has_content
            if end_by_step:
                sentence_spans_decoded.append((curr_start, i))
                curr_start = i + 1
                curr_has_content = False

        if curr_start < T and (not sentence_spans_decoded or sentence_spans_decoded[-1][1] < T - 1):
            sentence_spans_decoded.append((curr_start, T - 1))

        if not sentence_spans_decoded:
            sentence_spans_decoded = [(0, T - 1)]

        spans_local = sentence_spans_decoded
        step_texts_local = []
        

    if not spans_local:
        spans_local = [(0, max(0, T - 1))]

    if mat is None:
        return 0.5

    if step_cache is not None:
        prev_token_len = step_cache.get("token_len", 0)
        if prev_token_len > T_tokens:
            step_cache["texts"] = []
            step_cache["embs"] = None
            step_cache["logits_feats"] = None
            step_cache["token_len"] = 0
            prev_token_len = 0
    else:
        prev_token_len = 0

    S_total = len(spans_local)
    Fdim = len(used_feat_cols)

    logits_cached: Optional[np.ndarray] = None
    reuse_logits_upto = 0
    if use_incremental and step_cache is not None:
        cached_logits = step_cache.get("logits_feats", None)
        if isinstance(cached_logits, np.ndarray) and cached_logits.shape[1] == 3 * Fdim:
            reuse_logits_upto = min(cached_logits.shape[0], S_total)
            logits_cached = cached_logits

    step_texts_for_encode: List[str] = []
    new_step_logits_feats: List[List[float]] = []
    valid_idx = 0
    for idx, (st, ed) in enumerate(spans_local):
        if st > ed or st >= T:
            continue
        ed = min(ed, T - 1)
        span = mat[st:ed + 1, :]
        if span.size == 0:
            continue

        if use_incremental:
            step_texts_for_encode.append(step_texts_local[idx] if idx < len(step_texts_local) else "")
        else:
            prefix_before = tokenizer.decode(
                prompt_ids + gen_ids[:st],
                skip_special_tokens=False
            )
            prefix_after = tokenizer.decode(
                prompt_ids + gen_ids[:ed + 1],
                skip_special_tokens=False
            )
            step_texts_for_encode.append(prefix_after[len(prefix_before):])

        if logits_cached is not None and valid_idx < reuse_logits_upto:
            valid_idx += 1
            continue

        feats_step = []
        for j in range(Fdim):
            col = span[:, j]
            mean_v = float(col.mean())
            max_v = float(col.max())
            last_v = float(col[-1])
            feats_step.extend([mean_v, max_v, last_v])

        # norm_step_idx = idx / max(1, S_total - 1)
        # feats_step.append(norm_step_idx)
        new_step_logits_feats.append(feats_step)
        valid_idx += 1

    S_valid = valid_idx
    reuse_logits_upto = min(reuse_logits_upto, S_valid)

    parts: List[np.ndarray] = []
    if logits_cached is not None and reuse_logits_upto > 0:
        parts.append(logits_cached[:reuse_logits_upto, :])
    if new_step_logits_feats:
        parts.append(np.asarray(new_step_logits_feats, dtype=np.float32))

    if not parts:
        return 0.5

    X_logits = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    S = X_logits.shape[0]

    if use_incremental and step_cache is not None:
        step_cache["logits_feats"] = X_logits

    if step_cache is None:
        X_text = encode_step_texts(step_texts_for_encode, encoder)
        if X_text.shape[0] != S:
            return 0.5
    else:
        cached_texts: List[str] = step_cache.get("texts", [])
        cached_embs: Optional[np.ndarray] = step_cache.get("embs", None)

        if prev_token_len > T_tokens:
            step_cache["texts"] = []
            step_cache["embs"] = None
            step_cache["token_len"] = 0
            cached_texts = []
            cached_embs = None

        S_prev = len(cached_texts)
        if S_prev == 0:
            X_text = encode_step_texts(step_texts_for_encode, encoder)
            if X_text.shape[0] != S:
                return 0.5
            reuse_count = 0
        else:
            reuse_count = min(S_prev, S)
            reused = []
            if cached_embs is not None and cached_embs.shape[0] >= reuse_count:
                reused = [cached_embs[i] for i in range(reuse_count)]
            else:
                reused = []
                reuse_count = 0

            new_step_texts = step_texts_for_encode[reuse_count:]
            if new_step_texts:
                new_embs = encode_step_texts(new_step_texts, encoder)
                X_text = np.concatenate(
                    [np.stack(reused, axis=0)] + [new_embs] if reused else [new_embs],
                    axis=0
                )
            else:
                if reused:
                    X_text = np.stack(reused, axis=0)
                else:
                    X_text = encode_step_texts(step_texts_for_encode, encoder)

        step_cache["texts"] = list(step_texts_for_encode)
        step_cache["embs"] = X_text
        step_cache["token_len"] = T_tokens

        if S_prev == 0:
            print(f"[Cache Debug] : S={S}, token_len={T_tokens}")
        else:
            new_count = S - reuse_count
            print(f"[Cache Debug] : S_prev={S_prev}, S={S}, ={new_count}, token_len={T_tokens}")

        if X_text.shape[0] != S:
            return 0.5

    x_logits = torch.from_numpy(X_logits).unsqueeze(0).to(device)
    x_text   = torch.from_numpy(X_text).unsqueeze(0).to(device)
    lengths  = torch.tensor([S], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = step_model(x_logits, x_text, lengths)
        last_logit = logits[0, S - 1]
        p = torch.sigmoid(last_logit).item()

    return float(p)


# =====================
# =====================
def apply_repetition_penalty(logits: torch.Tensor, history_ids: List[int], penalty: float):
    if penalty is None or abs(penalty - 1.0) < 1e-6 or not history_ids:
        return logits
    logits = logits.clone()
    for tid in set(int(t) for t in history_ids):
        v = logits[0, tid]
        logits[0, tid] = torch.where(v > 0, v / penalty, v * penalty)
    return logits


def decode_piece_in_context(tokenizer, context_ids: List[int], tok_id: int) -> str:
    if tok_id is None:
        return ""
    return tokenizer.decode([tok_id], skip_special_tokens=True)

def piece_has_sentence_end(piece: str) -> bool:
    if not piece:
        return False
    p = piece.replace("\r\n", "\n")
    return STEP_TOKEN in p


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = None,
    generator: torch.Generator = None
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / max(1e-8, float(temperature))

    if top_p is not None and 0.0 < top_p < 1.0:
        probs = torch.softmax(scaled, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)

        keep_mask = cumprobs <= top_p
        keep_mask[..., 0] = True  # top1

        full_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
        full_mask.scatter_(dim=-1, index=sorted_idx, src=keep_mask)

        scaled = scaled.masked_fill(~full_mask, float("-inf"))

    if generator is None:
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    u = torch.rand(
        scaled.shape,
        dtype=scaled.dtype,
        device=scaled.device,
        generator=generator
    )
    u = u.clamp_(1e-6, 1.0 - 1e-6)
    g = -torch.log(-torch.log(u))

    idx = torch.argmax(scaled + g, dim=-1)
    return int(idx.item())



# =====================
# IO
# =====================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



from transformers.cache_utils import DynamicCache  #  DynamicCache， from transformers.cache_utils import Cache as DynamicCache
import torch


# def clone_past_cache(past, device=None, dtype=None):
#     return past
def clone_past_cache(past, device=None, dtype=None):
    dev = device
    if past is None:
        return None

    if hasattr(past, "to_legacy_cache"):
        legacy = past.to_legacy_cache()
        cloned_legacy = tuple(
            (k.detach().clone().contiguous().to(dev or k.device).to(dtype or k.dtype),
             v.detach().clone().contiguous().to(dev or v.device).to(dtype or v.dtype))
            for k, v in legacy
        )
        try:
            return type(past).from_legacy_cache(cloned_legacy)
        except Exception:
            try:
                from transformers.cache_utils import DynamicCache
                return DynamicCache.from_legacy_cache(cloned_legacy)
            except Exception:
                return cloned_legacy  #  legacy

    if isinstance(past, (list, tuple)) and len(past) > 0 and isinstance(past[0], (list, tuple)):
        return tuple(
            (k.detach().clone().contiguous().to(dev or k.device).to(dtype or k.dtype),
             v.detach().clone().contiguous().to(dev or v.device).to(dtype or v.dtype))
            for k, v in past
        )

    if hasattr(past, "_past_key_values"):
        legacy = past._past_key_values
        cloned_legacy = tuple(
            (k.detach().clone().contiguous().to(dev or k.device).to(dtype or k.dtype),
             v.detach().clone().contiguous().to(dev or v.device).to(dtype or v.dtype))
            for k, v in legacy
        )
        try:
            return type(past).from_legacy_cache(cloned_legacy)
        except Exception:
            try:
                from transformers.cache_utils import DynamicCache
                return DynamicCache.from_legacy_cache(cloned_legacy)
            except Exception:
                return cloned_legacy

    raise TypeError(f"Unrecognized past type: {type(past)}")


def gen_hash(gen: torch.Generator) -> str:
    dev = getattr(gen, "device", torch.device("cpu"))
    state = gen.get_state()
    if state.device.type != "cpu":
        state = state.cpu()
    buf = state.numpy().tobytes() + str(dev).encode()
    return hashlib.blake2b(buf, digest_size=8).hexdigest()

def clear_past_cache_from_anchor(anchor: Optional[Dict[str, Any]]):
    """anchorpkv，"""
    if anchor is not None and "pkv" in anchor:
        pkv = anchor["pkv"]
        if pkv is not None:
            try:
                def clear_tensor_recursive(obj):
                    """tensor"""
                    if isinstance(obj, torch.Tensor):
                        if obj.is_cuda:
                            obj_cpu = obj.cpu()
                            del obj_cpu
                        del obj
                    elif isinstance(obj, (tuple, list)):
                        for item in obj:
                            clear_tensor_recursive(item)
                    elif hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
                        for cache_list in [obj.key_cache, obj.value_cache]:
                            for layer_cache in cache_list:
                                if isinstance(layer_cache, torch.Tensor) and layer_cache.is_cuda:
                                    layer_cache_cpu = layer_cache.cpu()
                                    del layer_cache_cpu
                                del layer_cache
                
                clear_tensor_recursive(pkv)
            except Exception as e:
                print(f"past cache: {e}")
            finally:
                anchor["pkv"] = None

def clear_all_past_caches(latest_anchor, best_anchor, node_snapshots, sent_history):
    """past cache，"""
    clear_past_cache_from_anchor(latest_anchor)
    
    clear_past_cache_from_anchor(best_anchor)
    
    for snapshot in node_snapshots:
        if "pkv" in snapshot:
            clear_past_cache_from_anchor(snapshot)
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def update_speed_state(
    R_history: List[float],
    prev_state: str,
    fast_th: float = FAST_STEP_R_THRESHOLD,
    slow_th: float = SLOW_STEP_R_THRESHOLD,
    k_fast: int = 6,
    k_slow: int = 5,
    hysteresis: float = 0.1,
) -> str:
    """
    Update per-sentence speed state based on recent R values.

    States:
      - "MIDDLE": default / neutral
      - "FAST":   we believe current region is easy
      - "SLOW":   we believe current region is hard

    Logic (with simple hysteresis to avoid oscillation):
      - From MIDDLE:
          * if last k_fast R all >= fast_th → FAST
          * elif last k_slow R all <= slow_th → SLOW
          * else stay MIDDLE
      - From FAST:
          * if current R <= fast_th - hysteresis:
                - if last k_slow R all <= slow_th → SLOW
                - else → MIDDLE
            else stay FAST
      - From SLOW:
          * if current R >= slow_th + hysteresis:
                - if last k_fast R all >= fast_th → FAST
                - else → MIDDLE
            else stay SLOW
    """
    if not R_history:
        return prev_state

    r = R_history[-1]
    n = len(R_history)
    kf = min(k_fast, n)
    ks = min(k_slow, n)

    recent_fast = R_history[-kf:]
    recent_slow = R_history[-ks:]

    if prev_state == "MIDDLE":
        if n > k_fast and all(x >= fast_th for x in recent_fast):
            return "FAST"
        if n > k_slow and all(x <= slow_th for x in recent_slow):
            return "SLOW"
        return "MIDDLE"

    if prev_state == "FAST":
        # Only consider leaving FAST if R has dropped below fast_th - hysteresis
        if r <= fast_th - hysteresis:
            if ks > 0 and all(x <= slow_th for x in recent_slow):
                return "SLOW"
            return "MIDDLE"
        return "FAST"

    if prev_state == "SLOW":
        # Only consider leaving SLOW if R has risen above slow_th + hysteresis
        if r >= slow_th + hysteresis:
            if kf > 0 and all(x >= fast_th for x in recent_fast):
                return "FAST"
            return "MIDDLE"
        return "SLOW"

    # Fallback
    return prev_state

def run_one_item_with_think_control(
    question: str,
    mdl,
    tok,
    r_model,
    encoder,
    feat_cols,
    rank_like_cols,
    zstats,
    device: torch.device,
    think_len: int,           # ：<think>  token （ prompt）
    answer_len: int,          # ： token 
    temperature: float = 0.6,
    repetition_penalty: float = 1.0,
):
    """
    Two-stage generation with online R detector:

    Stage 1 (thinking):
      - Start from a prompt ending with "<think>\\n".
      - Generate token-by-token, applying the R-detector and speed states:
            {"MIDDLE", "FAST", "SLOW", "SKIP"}.
      - Insert [FAST_STEP]/[SLOW_STEP]/[SKIP_STEP] tags only when the state changes.
      - Stop when:
            * the model itself outputs "</think>" (preferred), OR
            * the thinking token budget (think_len) is exhausted (fallback).

    Stage 2 (answer):
      - Take the full context (prompt + all Stage 1 tokens) as input.
      - Run a separate generate(max_new_tokens=answer_len) WITHOUT R-detector.
      - Return the concatenation of thinking + answer as final text.
    """

    user_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        "During your thinking, you may see the following tags:\n"
        "[FAST_STEP] means the current step seems easy; keep your reasoning brief and avoid unnecessary details.\n"
        "[SLOW_STEP] means the current step seems difficult; please perform detailed reasoning.\n"
        "[MIDDLE_STEP] means the current step has moderate difficulty; please resume normal step-by-step reasoning.\n\n"
        "[SKIP_STEP] means this step is too difficult and further detailed expansion is not very helpful,please summarize the existing reasoning, make a reasonable guess for the conclusion, and then quickly output the final answer.\n\n "
        f"{question}\n"
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    input_ids = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=True, 
    ).to(device)
    inputs = {"input_ids": input_ids}
    prompt_ids = inputs["input_ids"][0].tolist()

    end_think_ids = tok.encode("</think>", add_special_tokens=False)
    end_think_len = len(end_think_ids)

    FAST_TAG_TEXT = "[FAST_STEP] "
    SLOW_TAG_TEXT = "[SLOW_STEP] "
    SKIP_TAG_TEXT = "[SKIP_STEP] "
    MIDDLE_TAG_TEXT = "[MIDDLE_STEP] " 

    fast_tag_ids = tok.encode(FAST_TAG_TEXT, add_special_tokens=False)
    slow_tag_ids = tok.encode(SLOW_TAG_TEXT, add_special_tokens=False)
    skip_tag_ids = tok.encode(SKIP_TAG_TEXT, add_special_tokens=False)
    middle_tag_ids = tok.encode(MIDDLE_TAG_TEXT, add_special_tokens=False)

    ignore_ids: set[int] = set(fast_tag_ids + slow_tag_ids + skip_tag_ids + middle_tag_ids)

    gen_ids: List[int] = []          # ，Stage 1  token
    R_gen_ids: List[int] = []        # R  token（）
    R_canonical_feats: Dict[str, List[float]] = init_canonical_feat_buffer()
    R_sent_history: List[Dict[str, Any]] = []

    R_curr_sent_tokens: List[int] = []
    R_curr_sent_text: str = ""
    R_curr_sent_start: int = 0

    R_step_cache: Dict[str, Any] = {
        "token_len": 0,
        "texts": [],
        "embs": None,
        "logits_feats": None,
    }

    R_history: List[float] = []
    mode_history: List[str] = []   # （MIDDLE/FAST/SLOW/SKIP）
    speed_state: str = "MIDDLE"    # 
    slow_locked: bool = False      #  SKIP  SLOW

    SKIP_WINDOW = 35   #  R  SKIP
    SKIP_R_TH   = 0.15

    first_skip_seen = False             #  SKIP
    steps_since_last_skip = 0          #  SKIP（ or ）
    SKIP_REPEAT_INTERVAL = 30          #  SKIP 

    total_token_consumed = 0
    done_think = False

    running_ctx_ids_for_piece = inputs["input_ids"][0].tolist()
    curr_sentence_has_content = False
    forced_token_ids: List[int] = []  # label token queue

    def last_matches_end_think(recent_tokens: List[int]) -> bool:
        if end_think_len == 0 or len(recent_tokens) < end_think_len:
            return False
        return recent_tokens[-end_think_len:] == end_think_ids

    # prefill prompt
    with torch.no_grad():
        out_prefill = mdl(**inputs, use_cache=True, return_dict=True)
    past = out_prefill.past_key_values

    prev_tok = inputs["input_ids"][:, -1:]

    while (not done_think) and (len(gen_ids) < think_len):
            with torch.no_grad():
                out = mdl(
                    input_ids=prev_tok,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :]

            # build current-step top-k pack (replaces RawLogitsTap)
            with torch.no_grad():
                topk_vals, topk_idx = torch.topk(
                    logits,
                    k=min(K_TOP, logits.shape[-1]),
                    dim=-1,
                )
            pack = {
                "topk_vals": topk_vals.detach().to("cpu", non_blocking=True),
                "topk_idx": topk_idx.detach().to("cpu", non_blocking=True),
            }

            if forced_token_ids:
                next_id = int(forced_token_ids.pop(0))
            else:
                logits = apply_repetition_penalty(logits, gen_ids, repetition_penalty)
                next_id = sample_from_logits(
                    logits,
                    temperature=temperature,
                    top_p=0.95,
                    generator=None,
                )

            next_id = int(next_id)
            gen_ids.append(next_id)
            total_token_consumed += 1

            piece = decode_piece_in_context(tok, running_ctx_ids_for_piece, next_id)
            running_ctx_ids_for_piece = running_ctx_ids_for_piece + [next_id]

            _payload = piece.replace("\r\n", "\n").replace("\n", "")
            if _payload.strip() != "":
                curr_sentence_has_content = True

            prev_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)

            if last_matches_end_think(gen_ids):
                print("[THINK] Model produced </think>, stop Stage 1.")
                done_think = True
                break

            if tok.eos_token_id is not None and next_id == int(tok.eos_token_id):
                print("[THINK] Hit EOS before </think>, stop Stage 1.")
                done_think = True
                break

            is_ignored = next_id in ignore_ids

            if not is_ignored:
                if len(R_curr_sent_tokens) == 0:
                    R_curr_sent_start = len(R_gen_ids)

                R_gen_ids.append(next_id)
                R_curr_sent_tokens.append(next_id)
                R_curr_sent_text += piece

                if pack is not None:
                    update_canonical_feats_incremental(
                        canonical_feats=R_canonical_feats,
                        pack=pack,
                        tok_id=next_id,
                        zwin_for_logp=ONLINE_ZWIN_FOR_LOGP,
                    )
            else:
                R_curr_sent_text += piece

            end_by_step = piece_has_sentence_end(piece) and curr_sentence_has_content
            if not end_by_step:
                continue

            curr_sentence_has_content = False

            if len(R_curr_sent_tokens) > 0:
                R_sent_history.append({
                    "tokens": list(R_curr_sent_tokens),
                    "start_gen_len": int(R_curr_sent_start),
                    "text": R_curr_sent_text,
                })
                R_curr_sent_tokens = []
                R_curr_sent_text = ""
                R_curr_sent_start = len(R_gen_ids)
            else:
                R_curr_sent_text = ""
                R_curr_sent_start = len(R_gen_ids)

            if not R_gen_ids:
                R_curr = 0.5
            else:
                try:
                    R_sentence_spans, R_step_texts = build_sentence_spans_and_texts(R_sent_history)

                    R_curr = r_prob_for_prefix(
                        step_model=r_model,
                            encoder=encoder,
                            tokenizer=tok,
                            prompt_ids=prompt_ids,
                            bucket=[],
                            gen_ids=R_gen_ids,
                        feat_cols=feat_cols,
                        zstats=zstats,
                        rank_like_cols=rank_like_cols,
                        device=device,
                        zwin_for_logp=ONLINE_ZWIN_FOR_LOGP,
                        step_cache=R_step_cache,
                        canonical_feats=R_canonical_feats,
                        sentence_spans=R_sentence_spans,
                        step_texts_external=R_step_texts,
                    )
                except Exception as e:
                    print(f"[WARN] r_prob_for_prefix failed, fallback to 0.5: {e}")
                    R_curr = 0.5

            R_curr = float(R_curr)
            R_history.append(R_curr)


            if first_skip_seen:
                steps_since_last_skip += 1

            prev_for_update = speed_state
            if prev_for_update not in ("MIDDLE", "FAST", "SLOW"):
                prev_for_update = "MIDDLE"

            base_state = update_speed_state(
                R_history=R_history,
                prev_state=prev_for_update,
                fast_th=FAST_STEP_R_THRESHOLD,
                slow_th=SLOW_STEP_R_THRESHOLD,
                k_fast=6,
                k_slow=5,
                hysteresis=0.1,
            )

            new_state = base_state

            if slow_locked and new_state == "SLOW":
                new_state = "MIDDLE"

            if (speed_state == "SLOW") and (len(R_history) >= SKIP_WINDOW):
                recent = R_history[-SKIP_WINDOW:]
                if all(r_val < SKIP_R_TH for r_val in recent):
                    new_state = "SKIP"
                    slow_locked = True
                    first_skip_seen = True           # ★  SKIP
                    steps_since_last_skip = 0  
                    print(f"[MODE] Enter SKIP state: last {SKIP_WINDOW} R all < {SKIP_R_TH}")


            if first_skip_seen and steps_since_last_skip >= SKIP_REPEAT_INTERVAL and new_state != "SKIP":
                new_state = "SKIP"
                steps_since_last_skip = 0
                print(f"[MODE] Periodic SKIP: force SKIP every {SKIP_REPEAT_INTERVAL} sentences after first SKIP")

            mode_history.append(new_state)
            sent_idx = len(R_history)
            print(f"[R-Detector] sentence {sent_idx}, R = {R_curr:.4f}, state = {new_state}")

            if new_state != speed_state:
                if new_state == "FAST":
                    forced_token_ids.extend(int(t) for t in fast_tag_ids)
                    print(f"[TAG] Enter FAST → FAST_STEP (R={R_curr:.4f})")
                elif new_state == "SLOW":
                    forced_token_ids.extend(int(t) for t in slow_tag_ids)
                    print(f"[TAG] Enter SLOW → SLOW_STEP (R={R_curr:.4f})")
                elif new_state == "SKIP":
                    forced_token_ids.extend(int(t) for t in skip_tag_ids)
                    print(f"[TAG] Enter SKIP → SKIP_STEP (R={R_curr:.4f})")
                elif new_state == "MIDDLE" and speed_state in ("FAST", "SLOW"):
                    forced_token_ids.extend(int(t) for t in middle_tag_ids)
                    print(f"[TAG] Enter MIDDLE → MIDDLE_STEP (R={R_curr:.4f})")
                speed_state = new_state

    ANSWER_PROMPT_TEXT = "\n\nthe answer is option"
    answer_prompt_ids = tok.encode(ANSWER_PROMPT_TEXT, add_special_tokens=False)


    for tid in answer_prompt_ids:
        tid = int(tid)
        with torch.no_grad():
            out = mdl(
                input_ids=torch.tensor([[tid]], dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        gen_ids.append(tid)
        total_token_consumed = total_token_consumed +1

        piece = decode_piece_in_context(tok, running_ctx_ids_for_piece, tid)
        running_ctx_ids_for_piece = running_ctx_ids_for_piece + [tid]

    answer_tokens = 0
    max_answer_tokens = answer_len  # 
    done_answer = False

    prev_tok = torch.tensor([[gen_ids[-1]]], dtype=torch.long, device=device)

    while (not done_answer) and (answer_tokens < max_answer_tokens):
        with torch.no_grad():
            out = mdl(
                input_ids=prev_tok,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        logits = apply_repetition_penalty(logits, gen_ids, repetition_penalty)
        next_id = sample_from_logits(
            logits,
            temperature=temperature,
            top_p=0.95,
            generator=None,
        )
        total_token_consumed = total_token_consumed +1
        next_id = int(next_id)

        gen_ids.append(next_id)
        answer_tokens += 1

        piece = decode_piece_in_context(tok, running_ctx_ids_for_piece, next_id)
        running_ctx_ids_for_piece = running_ctx_ids_for_piece + [next_id]

        if "\n" in piece:
            print(f"[ANSWER] Newline detected in piece={repr(piece)}; stop answer decoding.")
            done_answer = True
            break

        if tok.eos_token_id is not None and next_id == int(tok.eos_token_id):
            print("[ANSWER] Hit EOS while generating answer; stop.")
            done_answer = True
            break

        prev_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)

    total_gen_ids = gen_ids
    full_text = tok.decode(total_gen_ids, skip_special_tokens=True).strip()

    return {
        "generated_text": full_text,
        "gen_ids": total_gen_ids,
        "R_history": R_history,
        "mode_history": mode_history,  # records "MIDDLE"/"FAST"/"SLOW"/"SKIP" per sentence
        "total_token_consumed": total_token_consumed,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="../qwen3-4b")
    ap.add_argument("--input", type=str, default=INPUT_DATA_FILE)
    ap.add_argument("--out", type=str, default=OUT_JSONL)
    ap.add_argument(
        "--r-ckpt",
        type=str,
        default="step_seq_prm_gte_small_logits_gru_best_psr2.pt",
        help="2.0 StepSeqPRM_GRU ",
    )
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    mdl.eval()

    print(f"[R-Model v2]  zstats: {ZSTATS_PATH}")
    with open(ZSTATS_PATH, "r", encoding="utf-8") as f:
        zstats = json.load(f)

    feat_cols = BASE_FEAT_COLS[:]
    rank_like = set(RANK_LIKE_COLS)
    print(f"[R-Model v2] ： {feat_cols}")
    print(f"[R-Model v2] rank-like ： {rank_like}")

    enc_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[R-Model v2]  gte-small : {GTE_SMALL_PATH} (device={enc_device})")

    word_embedding_model = models.Transformer(
        GTE_SMALL_PATH,
        max_seq_length=512,
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    encoder = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device=enc_device,
    )

    d_text = encoder.get_sentence_embedding_dimension()
    d_logits = 3 * len(feat_cols)
    print(f"[R-Model v2] d_logits={d_logits}, d_text={d_text}")

    r_model = StepSeqPRM_GRU(
        d_logits=d_logits,
        d_text=d_text,
        d_model=128,
        num_layers=1,
        dropout=0.1,
    ).to(device)

    state = torch.load(args.r_ckpt, map_location=device)
    r_model.load_state_dict(state, strict=True)
    r_model.eval()
    print(f"[R-Model v2]  {args.r_ckpt}  2.0 GRU ")

    rows = read_jsonl(args.input)
    for idx, row in enumerate(tqdm(rows, desc="R-Detect w/ Think Control")):
        question = row.get("question") or row.get("problem") or ""
        standard_answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        think_len  = int(args.max_new_tokens * 0.9)
        answer_len = args.max_new_tokens- think_len

        res = run_one_item_with_think_control(
            question=question,
            mdl=mdl,
            tok=tok,
            r_model=r_model,
            encoder=encoder,
            feat_cols=feat_cols,
            rank_like_cols=rank_like,
            zstats=zstats,
            device=device,
            think_len=think_len,
            answer_len=answer_len,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )

        append_jsonl(args.out, [{
            "event": "item_done",
            "ts": time.time(),
            "index": idx,
            "question": (question or "")[:2000],
            "generated_answer": res["generated_text"],
            "generated_answer_id": [int(t) for t in res["gen_ids"]],
            "R_history": res["R_history"],
            "mode_history": res.get("mode_history", []),
            "standard_answer": standard_answer,
            "final_chain_tokens": int(res["total_token_consumed"]),
            "total_token_consumed": int(res["total_token_consumed"]),
        }])

        print_cuda_mem()

if __name__ == "__main__":
    setup_seed_global(SEED)
    main()
