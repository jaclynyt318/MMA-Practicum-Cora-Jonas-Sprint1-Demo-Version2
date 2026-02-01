# src/scoring/scoring.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .constants import REQUIRED_FIELDS
from .features import build_features, model_feature_columns
from .model_io import load_model
from .schema import apply_mapping, validate_and_coerce


def risk_tier_from_prob(p: pd.Series) -> pd.Series:
    # Simple fixed thresholds (presentation-friendly)
    return pd.Series(
        np.where(p >= 0.50, "High", np.where(p >= 0.35, "Medium", "Low")),
        index=p.index,
    )


def churn_timeline_from_prob(p: pd.Series) -> pd.Series:
    # Demo bucket mapping — can be tuned later
    return pd.Series(
        np.where(p >= 0.50, "0–90 days", np.where(p >= 0.35, "3–6 months", "6–12 months")),
        index=p.index,
    )


def driver_rules(row: pd.Series) -> Tuple[list[str], list[str]]:
    drivers: list[str] = []
    actions: list[str] = []

    if row.get("usage_drop_flag", 0) == 1:
        drivers.append("Usage decline")
        actions.append("Run re-engagement campaign + in-product training")

    if row.get("tickets_spike_flag", 0) == 1 or row.get("tickets_pct_change", 0) > 0.3:
        drivers.append("Support pressure rising")
        actions.append("Proactive support outreach + escalation review")

    if row.get("contract_ending_soon_flag", 0) == 1:
        drivers.append("Contract ending soon")
        actions.append("Start renewal outreach and confirm success plan")

    if row.get("downsell_flag", 0) == 1:
        drivers.append("Commercial contraction")
        actions.append("Commercial check-in: seats/value alignment")

    if not drivers:
        drivers = ["No dominant trigger (monitor)"]
        actions = ["Continue monitoring; reassess if new signals appear"]

    return drivers[:3], actions[:3]


def score_dataframe(
    df_input: pd.DataFrame,
    mapping: Dict[str, str],
    model_path=None,
) -> pd.DataFrame:
    """
    Core reusable scoring function.

    df_input: raw user-uploaded df (any schema)
    mapping: internal_name -> user_column_name (DISCO-style mapping results)
    return: df_output (account-level) with churn_probability, tiers, drivers, actions
    """
    # 1) mapping + schema coercion
    mapped = apply_mapping(df_input, mapping=mapping)
    schema_res = validate_and_coerce(mapped, required_fields=REQUIRED_FIELDS)
    if schema_res.missing_required:
        raise ValueError(f"Missing required fields after mapping: {schema_res.missing_required}")

    # 2) feature engineering
    fe = build_features(schema_res.df)

    # 3) model inference
    model = load_model(model_path) if model_path is not None else load_model()
    feat_cols = model_feature_columns()
    X = fe[feat_cols].copy()

    proba = model.predict_proba(X)[:, 1]
    out = fe.copy()
    out["churn_probability"] = proba
    out["risk_score"] = (out["churn_probability"] * 100).round(0).astype(int)
    out["risk_tier"] = risk_tier_from_prob(out["churn_probability"])
    out["churn_timeline"] = churn_timeline_from_prob(out["churn_probability"])

    # 4) drivers + actions (rule-based, explainable)
    driver_texts = []
    action_texts = []
    for _, r in out.iterrows():
        d, a = driver_rules(r)
        driver_texts.append(", ".join(d))
        action_texts.append(", ".join(a))

    out["top_drivers"] = driver_texts
    out["recommended_actions"] = action_texts

    # 5) Return a clean output view (keep context columns if present)
    keep_context = [c for c in ["plan_tier", "industry", "company_size", "seats_current", "arr_current"] if c in out.columns]
    output_cols = (
        ["account_id"] + keep_context +
        ["churn_probability", "risk_score", "risk_tier", "churn_timeline", "top_drivers", "recommended_actions"]
    )

    # Ensure account_id is present
    return out[output_cols].copy()
