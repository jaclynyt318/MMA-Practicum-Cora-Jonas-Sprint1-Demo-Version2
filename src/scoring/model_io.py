# src/scoring/model_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

DEFAULT_MODEL_PATH = Path("models") / "churn_risk_model.joblib"


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first: python -m src.training.train_model"
        )
    return joblib.load(model_path)


def save_model(model, model_path: Path = DEFAULT_MODEL_PATH):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
