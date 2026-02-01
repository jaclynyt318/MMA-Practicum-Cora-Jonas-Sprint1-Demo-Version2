# src/scoring/constants.py
from __future__ import annotations

# Internal field names that our scoring expects AFTER mapping.
REQUIRED_FIELDS = [
    "account_id",
    "plan_tier",
    "seats_current",
    "arr_current",
    "usage_count_current",
    "usage_count_prev",
    "tickets_opened_current",
    "tickets_opened_prev",
    # 可选：days_to_contract_end_current, avg_satisfaction_current
]


OPTIONAL_FIELDS = [
    "industry",
    "company_size",
    "mrr_current",
    "mrr_prev",
    "subscription_end_in_current_period",  # 0/1
]

# Human-friendly UI labels (centralized for consistent display).
UI_LABELS = {
    "account_id": "Account ID",
    "plan_tier": "Plan Tier",
    "industry": "Industry",
    "company_size": "Company Size",

    "seats_current": "Seats (Current)",
    "seats_prev": "Seats (Previous)",
    "arr_current": "ARR (Current)",
    "arr_prev": "ARR (Previous)",
    "mrr_current": "MRR (Current)",
    "mrr_prev": "MRR (Previous)",

    "usage_count_current": "Usage Count (Current)",
    "usage_count_prev": "Usage Count (Previous)",
    "tickets_opened_current": "Tickets Opened (Current)",
    "tickets_opened_prev": "Tickets Opened (Previous)",
    "avg_satisfaction_current": "Avg Satisfaction (Current)",
    "days_to_contract_end_current": "Days to Contract End (Current)",

    "churn_probability": "Churn Probability (0–1)",
    "risk_score": "Risk Score (0–100)",
    "risk_tier": "Risk Tier",
    "churn_timeline": "Churn Timeline",
    "top_drivers": "Top Drivers",
    "recommended_actions": "Recommended Actions",
}
