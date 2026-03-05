from pathlib import Path

import pandas as pd

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

NUMERIC_FEATURES = ["promo", "price", "competitor_price", "stockout"]


def drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        train_mean = float(train_df[col].mean())
        test_mean = float(test_df[col].mean())
        denom = abs(train_mean) + 1e-6
        shift = abs(test_mean - train_mean) / denom
        rows.append(
            {
                "feature": col,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "relative_shift": shift,
                "drift_flag": shift > threshold,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Missing train/test files. Run data pipeline first.")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    report = drift_report(train_df, test_df)
    print(report.to_string(index=False))

    if report["drift_flag"].any():
        print("\nWarning: Drift detected in one or more features.")
    else:
        print("\nNo significant drift detected.")


if __name__ == "__main__":
    main()