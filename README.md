# AI Demand Forecasting MLOps MVP

A production-style forecasting starter project with:
- Data generation and ingestion
- Model training + experiment tracking (MLflow)
- FastAPI inference service
- Data drift checks
- Scheduled retraining flow (Prefect)

## Project Structure

```text
mlops_forecasting/
  app.py
  data/
    raw/
    processed/
  models/
  src/
    data_pipeline.py
    train.py
    predict.py
    drift.py
    flow.py
    api.py
  requirements.txt
  requirements-train.txt
  .vercelignore
  README.md
```

## 1) Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements-train.txt
```

## 2) Generate synthetic data

```bash
python src/data_pipeline.py --generate
```

This creates:
- `data/raw/demand.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`

## 3) Train model

```bash
python src/train.py
```

Outputs:
- model: `models/forecast_model.pkl`
- metrics: printed + logged to MLflow

Optional MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## 4) Run drift check

```bash
python src/drift.py
```

## 5) Start inference API

```bash
uvicorn src.api:app --reload --port 8000
```

POST sample:

```json
{
  "date": "2026-03-10",
  "promo": 1,
  "price": 19.5,
  "competitor_price": 20.2,
  "stockout": 0
}
```

## 6) Run orchestrated retraining flow

```bash
python src/flow.py
```

This runs: ingest -> train -> drift check.

## 7) Deploy on Vercel

This repo includes:
- `app.py` as the Vercel Python function entrypoint
- `.vercelignore` to skip local-only heavy folders
- slim `requirements.txt` for inference-only runtime packages

### Deploy steps

```bash
npm i -g vercel
vercel login
vercel
vercel --prod
```

### Required files for deployment

- `models/forecast_model.pkl` must exist in the project
- `src/static/index.html` is served at `/`
- API endpoints:
  - `POST /chat`
  - `POST /predict`
  - `GET /health`

## Next improvements

1. Replace synthetic data with your real dataset source.
2. Add model registry + stage transitions in MLflow.
3. Add unit tests + CI/CD.
4. Deploy API with Docker + cloud runtime.
