from prefect import flow, task

try:
    from src.data_pipeline import RAW_PATH, build_datasets, generate_synthetic_data
    from src.drift import main as run_drift_check
    from src.train import main as run_training
except ImportError:
    from data_pipeline import RAW_PATH, build_datasets, generate_synthetic_data
    from drift import main as run_drift_check
    from train import main as run_training


@task
def ingest_task() -> None:
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_data().to_csv(RAW_PATH, index=False)
    build_datasets()


@task
def train_task() -> None:
    run_training()


@task
def drift_task() -> None:
    run_drift_check()


@flow(name="demand-forecast-retrain")
def retrain_flow() -> None:
    ingest_task()
    train_task()
    drift_task()


if __name__ == "__main__":
    retrain_flow()
