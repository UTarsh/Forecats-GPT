import datetime as dt
import math
from pathlib import Path

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
# In production this saves repeated disk I/O on every request.
_cache: dict = {"mode": "uninitialized", "reason": None, "model": None}


def _feature_row(record: dict) -> list[float]:
    d = dt.date.fromisoformat(str(record["date"]))
    return [
        float(record["promo"]),
        float(record["price"]),
        float(record["competitor_price"]),
        float(record["stockout"]),
        float(d.weekday()),
        float(d.month),
        float(d.timetuple().tm_yday),
    ]


def _heuristic_predict(record: dict) -> float:
    """
    Lightweight fallback used on constrained runtimes (e.g. Vercel).
    Mirrors the synthetic data generation dynamics so forecasts stay reasonable
    even when heavy ML dependencies are unavailable.
    """
    d = dt.date.fromisoformat(str(record["date"]))
    day_of_week = d.weekday()
    day_of_year = d.timetuple().tm_yday

    weekly = 12.0 * math.sin(2.0 * math.pi * day_of_week / 7.0)
    yearly = 18.0 * math.sin(2.0 * math.pi * day_of_year / 365.0)

    demand = (
        180.0
        + weekly
        + yearly
        + 30.0 * float(record["promo"])
        - 5.5 * float(record["price"])
        + 3.5 * float(record["competitor_price"])
        - 85.0 * float(record["stockout"])
    )
    return max(0.0, demand)


def load_model() -> None:
    if not MODEL_PATH.exists():
        _cache["mode"] = "heuristic"
        _cache["reason"] = f"Model file not found at {MODEL_PATH}."
        _cache["model"] = None
        return

    try:
        import joblib  # optional at runtime
    except Exception as e:
        _cache["mode"] = "heuristic"
        _cache["reason"] = f"joblib unavailable ({e.__class__.__name__})."
        _cache["model"] = None
        return

    try:
        _cache["model"] = joblib.load(MODEL_PATH)
        _cache["mode"] = "model"
        _cache["reason"] = None
    except Exception as e:
        _cache["mode"] = "heuristic"
        _cache["reason"] = f"Model load failed ({e.__class__.__name__})."
        _cache["model"] = None


def get_model():
    if _cache.get("mode") == "uninitialized":
        load_model()
    return _cache["model"]


def runtime_mode() -> str:
    if _cache.get("mode") == "uninitialized":
        load_model()
    return str(_cache.get("mode"))


def runtime_reason() -> str | None:
    if _cache.get("mode") == "uninitialized":
        load_model()
    return _cache.get("reason")


def predict_one(record: dict) -> float:
    model = get_model()
    if model is not None and _cache.get("mode") == "model":
        try:
            row = _feature_row(record)
            pred = float(model.predict([row])[0])
            return max(0.0, pred)
        except Exception as e:
            _cache["mode"] = "heuristic"
            _cache["reason"] = f"Model predict failed ({e.__class__.__name__})."
            _cache["model"] = None

    return _heuristic_predict(record)
