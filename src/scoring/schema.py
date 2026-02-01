# src/scoring/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SchemaResult:
    df: pd.DataFrame
    missing_required: List[str]
    warnings: List[str]


TRUE_STR = {"true", "t", "1", "yes", "y"}
FALSE_STR = {"false", "f", "0", "no", "n"}


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(TRUE_STR)] = 1
    out[s.isin(FALSE_STR)] = 0
    # If it's already numeric-ish, keep it
    num = pd.to_numeric(series, errors="coerce")
    out = out.fillna(num)
    return out.fillna(0).astype(int)


def coerce_numeric(series: pd.Series) -> pd.Series:
    # Robust numeric conversion. Non-numeric -> NaN (later imputed).
    return pd.to_numeric(series, errors="coerce")


def apply_mapping(
    df_input: pd.DataFrame,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """
    mapping: internal_name -> user_column_name
    Returns df with internal columns created.
    """
    df = df_input.copy()
    out = pd.DataFrame(index=df.index)

    for internal, user_col in mapping.items():
        if user_col is None or user_col == "":
            continue
        if user_col not in df.columns:
            continue
        out[internal] = df[user_col]

    # keep unmapped columns as attributes (pass-through)
    used_cols = {c for c in mapping.values() if c}
    passthrough = [c for c in df.columns if c not in used_cols]
    for c in passthrough:
        out[c] = df[c]

    return out


def validate_and_coerce(
    df_mapped: pd.DataFrame,
    required_fields: List[str],
) -> SchemaResult:
    warnings: List[str] = []
    missing = [c for c in required_fields if c not in df_mapped.columns]

    df = df_mapped.copy()

    # account_id must exist and be non-null-ish
    if "account_id" in df.columns:
        df["account_id"] = df["account_id"].astype(str)
        if df["account_id"].isna().any():
            warnings.append("Some account_id values are missing; they will be treated as 'nan' strings.")
    else:
        missing = list(set(missing + ["account_id"]))

    # Coerce known numeric columns if present
    numeric_like = [
        "seats_current", "seats_prev",
        "arr_current", "arr_prev",
        "mrr_current", "mrr_prev",
        "usage_count_current", "usage_count_prev",
        "tickets_opened_current", "tickets_opened_prev",
        "avg_satisfaction_current",
        "days_to_contract_end_current",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])

    # Coerce optional binary flag
    if "subscription_end_in_current_period" in df.columns:
        df["subscription_end_in_current_period"] = coerce_bool(df["subscription_end_in_current_period"])

    # plan_tier should be text
    if "plan_tier" in df.columns:
        df["plan_tier"] = df["plan_tier"].astype(str).str.strip().replace({"nan": ""})

    return SchemaResult(df=df, missing_required=missing, warnings=warnings)


def fingerprint_input(df: pd.DataFrame) -> str:
    # Used to avoid re-scoring when filters change.
    # Stable-ish hash based on shape + column names + first/last few ids.
    cols = "|".join(df.columns.astype(str).tolist())
    head_ids = ""
    if "account_id" in df.columns:
        head_ids = "|".join(df["account_id"].astype(str).head(5).tolist()) + "|" + "|".join(df["account_id"].astype(str).tail(5).tolist())
    key = f"{df.shape[0]}x{df.shape[1]}::{cols}::{head_ids}"
    return str(abs(hash(key)))
