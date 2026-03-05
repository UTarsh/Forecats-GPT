from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/forecast_model.pkl")

FEATURES = [
    "promo",
    "price",
    "competitor_price",
    "stockout",
    "day_of_week",
    "month",
    "day_of_year",
]

# Mutable container so we avoid `global`. Loaded once at startup.
# In production this saves ~50-200ms per call vs loading from disk every request.
_cache: dict = {}


def load_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python src/train.py")
    _cache["model"] = joblib.load(MODEL_PATH)


def get_model():
    if "model" not in _cache:
        load_model()
    return _cache["model"]


def prepare_features(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df[FEATURES]


def predict_one(record: dict) -> float:
    model = get_model()
    X = prepare_features(record)
    pred = float(model.predict(X)[0])
    return max(0.0, pred)  # demand can't be negative
