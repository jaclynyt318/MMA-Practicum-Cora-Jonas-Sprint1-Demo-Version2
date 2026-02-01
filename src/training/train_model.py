# src/training/train_model.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.scoring.features import build_features, model_feature_columns
from src.scoring.model_io import save_model
from .build_training_table import build_training_table


def main() -> None:
    df = build_training_table()
    # Build the SAME feature columns as scoring uses
    df_feat = build_features(df)

    label = df_feat["churned"].astype(int)
    feat_cols = model_feature_columns()
    X = df_feat[feat_cols].copy()

    categorical_cols = ["plan_tier"]
    numeric_cols = [c for c in feat_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]), categorical_cols),
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        penalty="l2",
        C=0.1,
        max_iter=1000,
        solver="lbfgs",
    )

    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X, label)

    save_model(pipeline)
    print("✅ Saved model to models/churn_risk_model.joblib")
    print(f"Training rows: {len(df_feat):,}, positives(churned): {int(label.sum()):,} ({label.mean():.3f})")


if __name__ == "__main__":
    main()
