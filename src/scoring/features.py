# src/scoring/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


# src/scoring/features.py

def model_feature_columns() -> list[str]:
    """
    Canonical feature list shared by training and scoring.

    IMPORTANT:
    - Must be computable from the single-table demo input
    - Must not contain all-NaN columns
    """
    return [
        # ===== core behavioral deltas =====
        "usage_delta",          # usage_current - usage_prev
        "tickets_delta",        # tickets_current - tickets_prev

        # ===== commercial level =====
        "seats_current",
        "arr_current",

        # ===== risk flags =====
        "usage_drop_flag",
        "subscription_end_in_quarter",
        "satisfaction_missing_flag",
        "contract_missing_flag",

        # ===== optional numeric signal =====
        "avg_satisfaction",

        # ===== categorical =====
        "plan_tier",
    ]


def _to_numeric(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(s, errors="coerce")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: np.nan})
    return num / den


def _ensure_col(df: pd.DataFrame, col: str, default) -> None:
    if col not in df.columns:
        df[col] = default


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal, robust feature set for churn risk demo.
    Assumptions:
      - Input df is an account-level snapshot table (one row per account per time unit),
        produced by build_training_table() for training or by upload/mapping for scoring.
    """
    out = df.copy()

    # -------------------------
    # If prev fields are NOT provided by user mapping,
    # default prev = current so delta becomes 0 (demo-friendly)
    # -------------------------
    # Important: We only do this when the column is missing OR entirely null.
    # This avoids "current - 0" fake deltas.
    def _default_prev_to_current(prev_col: str, cur_col: str) -> None:
        if prev_col not in df.columns:
            out[prev_col] = out.get(cur_col)
        else:
            # column exists; if it's all missing, treat as "not provided"
            if out[prev_col].isna().all():
                out[prev_col] = out.get(cur_col)

    _default_prev_to_current("seats_prev", "seats_current")
    _default_prev_to_current("arr_prev", "arr_current")
    _default_prev_to_current("mrr_prev", "mrr_current")

    # -------------------------
    # Ensure base columns exist
    # -------------------------
    # Commercial "current/prev" style fields are optional; we derive deltas if possible.
    # If your build_training_table already outputs *_delta fields, we just use them.
    for c in [
        "seats_current", "seats_prev",
        "arr_current", "arr_prev",
        "mrr_current", "mrr_prev",
        "seats", "arr_amount", "mrr_amount",
        "avg_satisfaction",
        "plan_tier",
        "usage_drop_flag",
        "subscription_end_in_quarter",
        "satisfaction_missing_flag",
        "contract_missing_flag",
        "churned",
    ]:
        _ensure_col(out, c, np.nan)

    # plan_tier: keep as string category-like
    out["plan_tier"] = out["plan_tier"].astype("object")

    # Flags: default to 0 if missing/NaN
    for flag in [
        "usage_drop_flag",
        "subscription_end_in_quarter",
        "satisfaction_missing_flag",
        "contract_missing_flag",
    ]:
        out[flag] = (
            pd.to_numeric(out[flag], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # Satisfaction numeric
    out["avg_satisfaction"] = _to_numeric(out.get("avg_satisfaction"))

    # -------------------------
    # Derive deltas if missing
    # -------------------------
    # seats_delta
    if "seats_delta" not in out.columns:
        a = _to_numeric(out["seats_current"])
        b = _to_numeric(out["seats_prev"])
        out["seats_delta"] = a - b


    # arr_delta
    if "arr_delta" not in out.columns:
        a = _to_numeric(out.get("arr_current"))
        b = _to_numeric(out.get("arr_prev"))
        if a.empty and b.empty:
            out["arr_delta"] = np.nan
        else:
            out["arr_delta"] = a.fillna(0) - b.fillna(0)

    # mrr_delta
    if "mrr_delta" not in out.columns:
        a = _to_numeric(out.get("mrr_current"))
        b = _to_numeric(out.get("mrr_prev"))
        if a.empty and b.empty:
            out["mrr_delta"] = np.nan
        else:
            out["mrr_delta"] = a.fillna(0) - b.fillna(0)

    # -------------------------
    # Percent changes (safe)
    # -------------------------
    # seats_pct_change
    if "seats_pct_change" not in out.columns:
        prev = _to_numeric(out.get("seats_prev"))
        out["seats_pct_change"] = safe_div(_to_numeric(out["seats_delta"]), prev)

    # arr_pct_change
    if "arr_pct_change" not in out.columns:
        prev = _to_numeric(out.get("arr_prev"))
        out["arr_pct_change"] = safe_div(_to_numeric(out["arr_delta"]), prev)

    # mrr_pct_change
    if "mrr_pct_change" not in out.columns:
        prev = _to_numeric(out.get("mrr_prev"))
        out["mrr_pct_change"] = safe_div(_to_numeric(out["mrr_delta"]), prev)

    # -------------------------
    # Final: ensure all model columns exist
    # -------------------------
    for c in model_feature_columns():
        if c not in out.columns:
            if c.endswith("_flag") or c in {
                "usage_drop_flag",
                "subscription_end_in_quarter",
                "satisfaction_missing_flag",
                "contract_missing_flag",
            }:
                out[c] = 0
            elif c == "plan_tier":
                out[c] = "Unknown"
            else:
                out[c] = np.nan

    # Make sure numeric feature columns are numeric
    num_like = [
        "seats_delta", "arr_delta", "mrr_delta",
        "seats_pct_change", "arr_pct_change", "mrr_pct_change",
        "avg_satisfaction",
    ]
    for c in num_like:
        out[c] = _to_numeric(out.get(c))

    return out
