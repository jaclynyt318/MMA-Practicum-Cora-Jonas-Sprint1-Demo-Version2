import os
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/derived"

FILES = {
    "accounts": "ravenstack_accounts.csv",
    "subs": "ravenstack_subscriptions.csv",
    "usage": "ravenstack_feature_usage.csv",
    "tickets": "ravenstack_support_tickets.csv",
    "churn": "ravenstack_churn_events.csv",
}

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def month_floor(series):
    # Convert to month start timestamp for grouping
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()

def main():
    ensure_dirs()

    accounts = pd.read_csv(os.path.join(RAW_DIR, FILES["accounts"]))
    subs = pd.read_csv(os.path.join(RAW_DIR, FILES["subs"]))
    usage = pd.read_csv(os.path.join(RAW_DIR, FILES["usage"]))
    tickets = pd.read_csv(os.path.join(RAW_DIR, FILES["tickets"]))
    churn = pd.read_csv(os.path.join(RAW_DIR, FILES["churn"]))

    # ---------- Accounts: signups by month ----------
    accounts["signup_month"] = month_floor(accounts["signup_date"])
    signups_by_month = (
        accounts.groupby("signup_month", as_index=False)
        .agg(signups=("account_id", "count"))
        .sort_values("signup_month")
    )
    signups_by_month.to_csv(os.path.join(OUT_DIR, "signups_by_month.csv"), index=False)

    # ---------- Subscriptions: MRR by month (simple operational metric) ----------
    # Note: subscriptions behaves like records/changes; we use start_date month as event month for EDA trend.
    subs["start_month"] = month_floor(subs["start_date"])
    subs["mrr_amount"] = pd.to_numeric(subs["mrr_amount"], errors="coerce").fillna(0)

    mrr_by_month = (
        subs.groupby("start_month", as_index=False)
        .agg(mrr=("mrr_amount", "sum"))
        .sort_values("start_month")
    )
    mrr_by_month.to_csv(os.path.join(OUT_DIR, "mrr_by_month.csv"), index=False)

    mrr_by_plan_by_month = (
        subs.groupby(["start_month", "plan_tier"], as_index=False)
        .agg(mrr=("mrr_amount", "sum"), records=("subscription_id", "count"))
        .sort_values(["start_month", "plan_tier"])
    )
    mrr_by_plan_by_month.to_csv(os.path.join(OUT_DIR, "mrr_by_plan_by_month.csv"), index=False)

    # Upgrades/downgrades trend
    subs["upgrade_flag"] = subs["upgrade_flag"].astype(bool)
    subs["downgrade_flag"] = subs["downgrade_flag"].astype(bool)
    plan_changes_by_month = (
        subs.groupby("start_month", as_index=False)
        .agg(
            upgrades=("upgrade_flag", "sum"),
            downgrades=("downgrade_flag", "sum"),
            subscription_records=("subscription_id", "count"),
        )
        .sort_values("start_month")
    )
    plan_changes_by_month.to_csv(os.path.join(OUT_DIR, "plan_changes_by_month.csv"), index=False)

    # Billing split
    billing_split = (
        subs["billing_frequency"].value_counts(dropna=False)
        .rename_axis("billing_frequency")
        .reset_index(name="count")
    )
    billing_split.to_csv(os.path.join(OUT_DIR, "billing_split.csv"), index=False)

    # ---------- Support tickets: tickets by month ----------
    tickets["submitted_month"] = month_floor(tickets["submitted_at"])
    tickets_by_month = (
        tickets.groupby("submitted_month", as_index=False)
        .agg(
            tickets=("ticket_id", "count"),
            escalations=("escalation_flag", "sum"),
        )
        .sort_values("submitted_month")
    )
    tickets_by_month.to_csv(os.path.join(OUT_DIR, "tickets_by_month.csv"), index=False)

    # Ticket priority distribution
    priority_dist = (
        tickets["priority"].value_counts(dropna=False)
        .rename_axis("priority")
        .reset_index(name="count")
    )
    priority_dist.to_csv(os.path.join(OUT_DIR, "ticket_priority_dist.csv"), index=False)

    # Satisfaction coverage
    satisfaction_coverage = pd.DataFrame([{
        "total_tickets": len(tickets),
        "satisfaction_non_null": int(tickets["satisfaction_score"].notna().sum()),
        "satisfaction_null": int(tickets["satisfaction_score"].isna().sum()),
        "satisfaction_coverage_rate": float(tickets["satisfaction_score"].notna().mean()),
    }])
    satisfaction_coverage.to_csv(os.path.join(OUT_DIR, "satisfaction_coverage.csv"), index=False)

    # ---------- Churn events: churn by month + reasons ----------
    churn["churn_month"] = month_floor(churn["churn_date"])
    churn_events_by_month = (
        churn.groupby("churn_month", as_index=False)
        .agg(
            churn_events=("churn_event_id", "count"),
            accounts_affected=("account_id", "nunique"),
            refunds_positive=("refund_amount_usd", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum()),
        )
        .sort_values("churn_month")
    )
    churn_events_by_month.to_csv(os.path.join(OUT_DIR, "churn_events_by_month.csv"), index=False)

    churn_reasons = (
        churn["reason_code"].value_counts(dropna=False)
        .rename_axis("reason_code")
        .reset_index(name="count")
    )
    churn_reasons.to_csv(os.path.join(OUT_DIR, "churn_reasons.csv"), index=False)

    reactivation = pd.DataFrame([{
        "total_churn_events": len(churn),
        "reactivation_true": int(churn["is_reactivation"].astype(bool).sum()),
        "reactivation_rate": float(churn["is_reactivation"].astype(bool).mean()),
    }])
    reactivation.to_csv(os.path.join(OUT_DIR, "reactivation_summary.csv"), index=False)

    # ---------- Feature usage: usage by month + top features ----------
    usage["usage_month"] = month_floor(usage["usage_date"])
    usage["usage_count"] = pd.to_numeric(usage["usage_count"], errors="coerce").fillna(0)
    usage["usage_duration_secs"] = pd.to_numeric(usage["usage_duration_secs"], errors="coerce").fillna(0)
    usage["error_count"] = pd.to_numeric(usage["error_count"], errors="coerce").fillna(0)
    usage["is_beta_feature"] = usage["is_beta_feature"].astype(bool)

    usage_by_month = (
        usage.groupby("usage_month", as_index=False)
        .agg(
            usage_records=("usage_id", "count"),
            total_usage_count=("usage_count", "sum"),
            total_duration_secs=("usage_duration_secs", "sum"),
            total_errors=("error_count", "sum"),
            beta_records=("is_beta_feature", "sum"),
        )
        .sort_values("usage_month")
    )
    usage_by_month.to_csv(os.path.join(OUT_DIR, "usage_by_month.csv"), index=False)

    top_features = (
        usage.groupby("feature_name", as_index=False)
        .agg(
            usage_records=("usage_id", "count"),
            total_usage_count=("usage_count", "sum"),
            total_duration_secs=("usage_duration_secs", "sum"),
            total_errors=("error_count", "sum"),
            beta_share=("is_beta_feature", "mean"),
        )
        .sort_values("total_usage_count", ascending=False)
        .head(15)
    )
    top_features.to_csv(os.path.join(OUT_DIR, "top_features.csv"), index=False)

    # ---------- Executive KPIs snapshot ----------
    latest_mrr_month = mrr_by_month["start_month"].max() if len(mrr_by_month) else None
    latest_mrr = float(mrr_by_month.loc[mrr_by_month["start_month"] == latest_mrr_month, "mrr"].sum()) if latest_mrr_month is not None else 0

    kpis = pd.DataFrame([{
        "accounts": int(accounts["account_id"].nunique()),
        "subscription_records": int(subs["subscription_id"].nunique()),
        "tickets": int(tickets["ticket_id"].nunique()),
        "churn_events": int(churn["churn_event_id"].nunique()),
        "latest_mrr_month": str(latest_mrr_month.date()) if latest_mrr_month is not None else "",
        "latest_mrr": latest_mrr,
    }])
    kpis.to_csv(os.path.join(OUT_DIR, "kpis_snapshot.csv"), index=False)

    print("✅ Derived EDA tables generated in data/derived/")

if __name__ == "__main__":
    main()
