from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/forecast_model.pkl")

# These are the input columns our model learns from.
# We never include 'date' directly — models can't do math on raw dates.
# Instead we extract day_of_week, month, day_of_year as numbers.
FEATURES = [
    "promo",
    "price",
    "competitor_price",
    "stockout",
    "day_of_week",
    "month",
    "day_of_year",
]
TARGET = "demand"


def evaluate(y_true, y_pred) -> dict:
    """
    Three standard regression metrics:
      MAE  — average absolute error (in same units as demand, i.e. units sold)
      RMSE — like MAE but penalizes large errors more heavily
      R2   — how much variance the model explains (1.0 = perfect, 0 = baseline mean)
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_random_forest(X_train, y_train, X_test, y_test) -> tuple:
    """
    RandomForest: builds 300 trees independently, then averages predictions.
    n_estimators = number of trees
    max_depth    = how many levels deep each tree can split
    n_jobs=-1    = use all CPU cores
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test))
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test) -> tuple:
    """
    XGBoost: builds trees sequentially. Each tree fixes the errors of the previous one.

    Key hyperparameters:
      n_estimators  = how many trees (boosting rounds)
      max_depth     = complexity per tree (3-6 is typical; too high = overfitting)
      learning_rate = how aggressively each tree corrects errors (lower = more careful)
      subsample     = fraction of training rows each tree sees (0.8 = 80%)
      colsample_bytree = fraction of features each tree sees (prevents overfitting)
    """
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,  # silence training logs
    )
    model.fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict(X_test))
    return model, metrics


def train_and_compare() -> dict:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # --- Train both models ---
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)

    # --- Log both runs to MLflow so you can compare them visually ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("demand-forecasting")

    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_params({"model": "RandomForestRegressor", "n_estimators": 300, "max_depth": 14})
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, "model")

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({
            "model": "XGBRegressor",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
        })
        mlflow.log_metrics(xgb_metrics)
        mlflow.sklearn.log_model(xgb_model, "model")

    # --- Save the better model (lower MAE wins) ---
    if xgb_metrics["mae"] < rf_metrics["mae"]:
        best_model, best_name = xgb_model, "XGBoost"
    else:
        best_model, best_name = rf_model, "RandomForest"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    return {"random_forest": rf_metrics, "xgboost": xgb_metrics, "winner": best_name}


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Train/test datasets not found. Run: python src/data_pipeline.py --generate"
        )

    results = train_and_compare()
    rf = results["random_forest"]
    xg = results["xgboost"]

    print("\n=== Model Comparison ===")
    print(f"{'Metric':<8} {'RandomForest':>14} {'XGBoost':>10}")
    print("-" * 36)
    print(f"{'MAE':<8} {rf['mae']:>14.3f} {xg['mae']:>10.3f}")
    print(f"{'RMSE':<8} {rf['rmse']:>14.3f} {xg['rmse']:>10.3f}")
    print(f"{'R2':<8} {rf['r2']:>14.3f} {xg['r2']:>10.3f}")
    print("-" * 36)
    print(f"\nWinner: {results['winner']}  (saved to {MODEL_PATH})")


if __name__ == "__main__":
    main()
