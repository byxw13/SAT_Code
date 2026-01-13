# preprocess_steps_gte_small_logits.py
# -*- coding: utf-8 -*-
"""
 train/test CSV  step-level ，：
- logits pooling （ XGBoost ）
- gte-small  embedding
-  y_hard（reward>0.4 ）
-  y_soft（ reward ∈ [0,1]）
- sample_ids（ Mean Per-Sample F1）
"""

import numpy as np
import pandas as pd
from typing import List

from sentence_transformers import SentenceTransformer, models

from train_transformer_model_tensor_prefix_version_ZengLiu_V5_2_gpu_log_4 import (
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    ANSWER_COL,
    TARGET_REWARD,
    BASE_FEAT_COLS,
    RANK_LIKE_COLS,
    safe_load_list,
    extract_base_arrays,
    load_token_ids_for_row,
    build_step_mapping_using_prm_method,
    zscore_fit,
    prepare_input,
    _prm_tokenizer,
)

TRAIN_NPZ_PATH = "step_train_gte_small_logits.npz"
TEST_NPZ_PATH  = "step_test_gte_small_logits.npz"

REWARD_THRESHOLD = 0.4

MAX_STEPS_PER_SAMPLE = None


def build_step_level_dataset_with_text(
    df: pd.DataFrame,
    feat_cols: List[str],
    zstats,
    encoder: SentenceTransformer,
    reward_threshold: float = REWARD_THRESHOLD,
    log_n: int = 3,
):
    """
    “” DataFrame，“ step ”：
      - X_logits:  logits pooling 
      - X_text:    gte-small embedding
      - y_hard:   0/1 （reward>threshold）
      - y_soft:   （reward ）
      - sample_ids:  df.index

    ：
      X_logits, X_text, y_hard, y_soft, sample_ids
    """
    X_logits_list = []
    X_text_list   = []
    y_hard_list   = []
    y_soft_list   = []
    sample_ids    = []

    debug_printed = 0
    total_steps   = 0

    for ridx, row in df.iterrows():
        rew = safe_load_list(row.get(TARGET_REWARD))
        if not rew or len(rew) == 0:
            continue
        reward = [float(x) for x in rew]
        reward = [min(1.0, max(0.0, x)) for x in reward]
        S = len(reward)

        ans = row.get(ANSWER_COL, "")
        if not isinstance(ans, str) or len(ans) == 0:
            continue

        seqs = extract_base_arrays(row, feat_cols)
        lens = [len(v) for v in seqs.values() if len(v) > 0]
        if not lens:
            continue
        T = min(lens)
        if T <= 0:
            continue

        Fdim = len(feat_cols)
        mat = np.zeros((T, Fdim), dtype=np.float32)
        for j, c in enumerate(feat_cols):
            a = seqs.get(c, np.asarray([], dtype=np.float32))
            if a.size == 0:
                continue
            arr = a[:T].astype(np.float32)
            if c in RANK_LIKE_COLS:
                arr = np.log1p(np.maximum(0.0, arr))
            m, s = zstats.get(c, (0.0, 1.0))
            s = 1.0 if s <= 1e-12 else s
            mat[:, j] = (arr - m) / s

        token_ids = load_token_ids_for_row(row, T_required=T)
        if token_ids is None or len(token_ids) == 0:
            continue

        mapped = build_step_mapping_using_prm_method(ans, token_ids)
        if len(mapped) == 0:
            continue

        _, steps, _ = prepare_input("", ans, _prm_tokenizer, "\n")
        if debug_printed < log_n and (len(mapped) != S or len(steps) != S):
            print(f"[DEBUG] row {ridx}: reward_len={S}, mapped_steps={len(mapped)}, steps_text={len(steps)}")
            debug_printed += 1

        num_steps_here = 0
        max_k = min(len(mapped), len(steps), S)

        prev_end = -1
        row_logits_feats = []
        row_soft_labels  = []
        row_hard_labels  = []
        row_step_texts   = []

        for k in range(max_k):
            end_pos = mapped[k]
            if end_pos >= T:
                continue
            start_pos = prev_end + 1
            if start_pos > end_pos:
                prev_end = end_pos
                continue

            span = mat[start_pos:end_pos + 1]  # [L_step, Fdim]
            if span.size == 0:
                prev_end = end_pos
                continue

            step_feats = []
            for j in range(Fdim):
                col = span[:, j]
                mean_v = float(col.mean())
                max_v  = float(col.max())
                last_v = float(col[-1])
                step_feats.extend([mean_v, max_v, last_v])

            norm_step_idx = k / max(1, max_k - 1)
            step_feats.append(norm_step_idx)

            r_soft = reward[k]
            y_hard = 1 if r_soft > reward_threshold else 0

            row_logits_feats.append(step_feats)
            row_soft_labels.append(r_soft)
            row_hard_labels.append(y_hard)
            row_step_texts.append(steps[k])

            prev_end = end_pos
            num_steps_here += 1

            if MAX_STEPS_PER_SAMPLE is not None and num_steps_here >= MAX_STEPS_PER_SAMPLE:
                break

        if num_steps_here == 0:
            continue

        embs = encoder.encode(
            row_step_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if embs.shape[0] != len(row_logits_feats):
            print(f"[WARN] embedding count mismatch on row {ridx}: embs={embs.shape[0]}, steps={len(row_logits_feats)}")
            continue

        X_logits_list.extend(row_logits_feats)
        X_text_list.extend(embs.tolist())
        y_soft_list.extend(row_soft_labels)
        y_hard_list.extend(row_hard_labels)
        sample_ids.extend([ridx] * num_steps_here)

        total_steps += num_steps_here

    if not X_logits_list:
        raise RuntimeError(" step-level ，/。")

    X_logits = np.asarray(X_logits_list, dtype=np.float32)
    X_text   = np.asarray(X_text_list,   dtype=np.float32)
    y_soft   = np.asarray(y_soft_list,   dtype=np.float32)
    y_hard   = np.asarray(y_hard_list,   dtype=np.int64)
    sample_ids = np.asarray(sample_ids,  dtype=np.int64)

    print(f"[Build-Step-Dataset] total_steps={total_steps}")
    print(f"  X_logits.shape={X_logits.shape}")
    print(f"  X_text.shape  ={X_text.shape}")
    print(f"  y_soft.shape  ={y_soft.shape}")
    print(f"  y_hard.shape  ={y_hard.shape}")
    pos_ratio = float((y_hard == 1).mean())
    print(f"[Label] good_step_ratio={pos_ratio:.4f}, bad_step_ratio={1.0 - pos_ratio:.4f}")

    return X_logits, X_text, y_hard, y_soft, sample_ids


def main():
    train_df = pd.read_csv(
        TRAIN_FILE_PATH,
        encoding="utf-8",
        encoding_errors="ignore",
        low_memory=False
    )
    print(f"[Data] loaded train df: {train_df.shape}")

    feat_cols = [c for c in BASE_FEAT_COLS if c in train_df.columns]
    if not feat_cols:
        raise RuntimeError(" BASE_FEAT_COLS。")
    print(f"[Feat] Using {len(feat_cols)} base feat cols: {feat_cols}")

    zstats = zscore_fit(train_df, feat_cols)

    device = "cuda"  # : "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Encoder] Loading gte-small from ./Gte-Small on device={device} ...")
    

    word_embedding_model = models.Transformer(
        "../Gte-Small",          # 
        max_seq_length=512
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    encoder = SentenceTransformer(
        modules=[word_embedding_model, pooling_model]
    )


    print("[Preprocess] Building step-level dataset for TRAIN (with gte-small + logits)...")
    X_tr_logits, X_tr_text, y_tr_hard, y_tr_soft, sid_tr = build_step_level_dataset_with_text(
        train_df, feat_cols, zstats, encoder, reward_threshold=REWARD_THRESHOLD
    )

    np.savez(
        TRAIN_NPZ_PATH,
        X_logits=X_tr_logits,
        X_text=X_tr_text,
        y_hard=y_tr_hard,
        y_soft=y_tr_soft,
        sample_ids=sid_tr,
        reward_threshold=REWARD_THRESHOLD,
    )
    print(f"[Save] TRAIN step-level dataset -> {TRAIN_NPZ_PATH}")

    try:
        test_df = pd.read_csv(
            TEST_FILE_PATH,
            encoding="utf-8",
            encoding_errors="ignore",
            low_memory=False
        )
        print(f"[Data] loaded test df: {test_df.shape}")

        print("[Preprocess] Building step-level dataset for TEST (with gte-small + logits)...")
        X_te_logits, X_te_text, y_te_hard, y_te_soft, sid_te = build_step_level_dataset_with_text(
            test_df, feat_cols, zstats, encoder, reward_threshold=REWARD_THRESHOLD
        )

        np.savez(
            TEST_NPZ_PATH,
            X_logits=X_te_logits,
            X_text=X_te_text,
            y_hard=y_te_hard,
            y_soft=y_te_soft,
            sample_ids=sid_te,
            reward_threshold=REWARD_THRESHOLD,
        )
        print(f"[Save] TEST step-level dataset -> {TEST_NPZ_PATH}")
    except FileNotFoundError:
        print("[Warn] TEST_FILE_PATH ， npz。")


if __name__ == "__main__":
    main()
