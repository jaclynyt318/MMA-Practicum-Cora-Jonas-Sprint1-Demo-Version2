# src/training/build_training_table.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data") / "raw"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / name)


def build_training_table(as_of_date: str | None = None) -> pd.DataFrame:
    accounts = _read("ravenstack_accounts.csv")
    subs = _read("ravenstack_subscriptions.csv")
    usage = _read("ravenstack_feature_usage.csv")
    tickets = _read("ravenstack_support_tickets.csv")
    churn = _read("ravenstack_churn_events.csv")

    # Parse dates
    accounts["signup_date"] = pd.to_datetime(accounts.get("signup_date"), errors="coerce")
    subs["start_date"] = pd.to_datetime(subs.get("start_date"), errors="coerce")
    subs["end_date"] = pd.to_datetime(subs.get("end_date"), errors="coerce")
    usage["usage_date"] = pd.to_datetime(usage.get("usage_date"), errors="coerce")
    tickets["submitted_at"] = pd.to_datetime(tickets.get("submitted_at"), errors="coerce")
    churn["churn_date"] = pd.to_datetime(churn.get("churn_date"), errors="coerce")

    # Decide as_of_date
    if as_of_date is None:
        # latest date seen in any event table (safe default)
        candidates = [
            subs["start_date"].max(),
            subs["end_date"].max(),
            usage["usage_date"].max(),
            tickets["submitted_at"].max(),
            churn["churn_date"].max(),
        ]
        as_of = pd.to_datetime(pd.Series(candidates).dropna().max())
    else:
        as_of = pd.to_datetime(as_of_date)

    # ---- label: churned at any point (from churn_events) ----
    churn_label = (
        churn.dropna(subset=["account_id"])
        .groupby("account_id", as_index=False)
        .agg(churned=("churn_event_id", "count"))
    )
    churn_label["churned"] = (churn_label["churned"] > 0).astype(int)

    # ---- subscriptions: last known plan/seats/arr ----
    subs = subs.dropna(subset=["account_id", "start_date"]).copy()
    subs = subs.sort_values(["account_id", "start_date"])
    last_sub = subs.groupby("account_id", as_index=False).tail(1)

    # derive ARR if missing
    if "arr_amount" not in last_sub.columns and "mrr_amount" in last_sub.columns:
        last_sub["arr_amount"] = pd.to_numeric(last_sub["mrr_amount"], errors="coerce") * 12.0

    sub_feat = last_sub[["account_id"]].copy()
    sub_feat["plan_tier"] = last_sub.get("plan_tier")
    sub_feat["seats_current"] = pd.to_numeric(last_sub.get("seats"), errors="coerce")
    sub_feat["arr_current"] = pd.to_numeric(last_sub.get("arr_amount"), errors="coerce")

    # contract days
    end_date = last_sub.get("end_date")
    end_date = pd.to_datetime(end_date, errors="coerce")
    sub_feat["days_to_contract_end_current"] = (end_date - as_of).dt.days

    # ---- usage windows: last 30d vs prior 30d ----
    usage = usage.dropna(subset=["subscription_id", "usage_date"]).copy()
    # Map subscription -> account
    sub_map = subs[["subscription_id", "account_id"]].drop_duplicates()
    usage = usage.merge(sub_map, on="subscription_id", how="left")

    usage["usage_count"] = pd.to_numeric(usage.get("usage_count"), errors="coerce").fillna(0)

    w1_start = as_of - pd.Timedelta(days=30)
    w0_start = as_of - pd.Timedelta(days=60)

    u_cur = usage[(usage["usage_date"] > w1_start) & (usage["usage_date"] <= as_of)]
    u_prev = usage[(usage["usage_date"] > w0_start) & (usage["usage_date"] <= w1_start)]

    u1 = u_cur.groupby("account_id", as_index=False).agg(usage_count_current=("usage_count", "sum"))
    u0 = u_prev.groupby("account_id", as_index=False).agg(usage_count_prev=("usage_count", "sum"))

    # ---- tickets windows: last 30d vs prior 30d ----
    tickets = tickets.dropna(subset=["account_id", "submitted_at"]).copy()
    t_cur = tickets[(tickets["submitted_at"] > w1_start) & (tickets["submitted_at"] <= as_of)]
    t_prev = tickets[(tickets["submitted_at"] > w0_start) & (tickets["submitted_at"] <= w1_start)]

    t1 = t_cur.groupby("account_id", as_index=False).agg(
        tickets_opened_current=("ticket_id", "count") if "ticket_id" in t_cur.columns else ("submitted_at", "count"),
        avg_satisfaction_current=("satisfaction_score", "mean") if "satisfaction_score" in t_cur.columns else ("submitted_at", lambda s: np.nan),
    )
    t0 = t_prev.groupby("account_id", as_index=False).agg(
        tickets_opened_prev=("ticket_id", "count") if "ticket_id" in t_prev.columns else ("submitted_at", "count"),
    )

    # ---- assemble ----
    base = accounts[["account_id", "industry"]].copy()
    out = (
        base
        .merge(sub_feat, on="account_id", how="left")
        .merge(u1, on="account_id", how="left")
        .merge(u0, on="account_id", how="left")
        .merge(t1, on="account_id", how="left")
        .merge(t0, on="account_id", how="left")
        .merge(churn_label[["account_id", "churned"]], on="account_id", how="left")
    )

    out["churned"] = out["churned"].fillna(0).astype(int)

    # If missing windows, treat as 0
    for c in ["usage_count_current", "usage_count_prev", "tickets_opened_current", "tickets_opened_prev"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0)

    return out
