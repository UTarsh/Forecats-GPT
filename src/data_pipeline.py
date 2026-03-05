import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/demand.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def generate_synthetic_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    base_price = 20 + rng.normal(0, 0.8, size=n_days)
    competitor_price = base_price + rng.normal(0.4, 0.9, size=n_days)
    promo = rng.binomial(1, 0.25, size=n_days)
    stockout = rng.binomial(1, 0.03, size=n_days)

    day_of_year = dates.dayofyear.values
    weekly = 12 * np.sin(2 * np.pi * (dates.dayofweek.values) / 7)
    yearly = 18 * np.sin(2 * np.pi * day_of_year / 365)

    demand = (
        180
        + weekly
        + yearly
        + 30 * promo
        - 5.5 * base_price
        + 3.5 * competitor_price
        - 85 * stockout
        + rng.normal(0, 7, size=n_days)
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "promo": promo,
            "price": base_price.round(2),
            "competitor_price": competitor_price.round(2),
            "stockout": stockout,
            "demand": np.clip(demand, 0, None).round(2),
        }
    )
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    return out


def build_datasets() -> None:
    df = pd.read_csv(RAW_PATH)
    df = feature_engineer(df)

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Saved train: {TRAIN_PATH} ({len(train_df)} rows)")
    print(f"Saved test:  {TEST_PATH} ({len(test_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic raw data")
    args = parser.parse_args()

    if args.generate:
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_data()
        df.to_csv(RAW_PATH, index=False)
        print(f"Generated raw data at {RAW_PATH} ({len(df)} rows)")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset missing: {RAW_PATH}. Run with --generate first.")

    build_datasets()


if __name__ == "__main__":
    main()