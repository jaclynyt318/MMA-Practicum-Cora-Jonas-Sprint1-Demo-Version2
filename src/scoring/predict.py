import pandas as pd
import joblib

from src.scoring.features import build_features, model_feature_columns


MODEL_PATH = "models/churn_risk_model.joblib"


def risk_tier_from_prob(p: float) -> str:
    if p >= 0.50:
        return "High"
    elif p >= 0.35:
        return "Medium"
    else:
        return "Low"


def score_accounts(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Input: single-table account-level data (uploaded by user)
    Output: risk table
    """
    df_feat = build_features(df_raw)

    feat_cols = model_feature_columns()
    X = df_feat[feat_cols]

    model = joblib.load(MODEL_PATH)
    probs = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "account_id": df_raw["account_id"],
        "churn_probability": probs,
        "risk_tier": [risk_tier_from_prob(p) for p in probs],
    })

    return out.sort_values("churn_probability", ascending=False)
