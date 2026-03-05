from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from src.predict import load_model, predict_one
    from src.predict import MODEL_PATH, runtime_mode, runtime_reason
    from src.chat import process_message
except ImportError:
    from predict import load_model, predict_one
    from predict import MODEL_PATH, runtime_mode, runtime_reason
    from chat import process_message


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()   # load trained model once at startup, fallback to heuristic if needed
    yield


app = FastAPI(
    title="Demand Forecast API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the chat UI from /static/
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── HTML chat UI ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Serve the chatbot UI."""
    return (_static_dir / "index.html").read_text(encoding="utf-8")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "model_exists": MODEL_PATH.exists(),
        "runtime_mode": runtime_mode(),  # "model" or "heuristic"
        "runtime_reason": runtime_reason(),
    }


# ── JSON prediction endpoint (for direct API use) ─────────────────────────────

class ForecastRequest(BaseModel):
    date: str = Field(..., examples=["2026-03-10"], description="ISO date YYYY-MM-DD")
    promo: int = Field(..., ge=0, le=1, description="1 if promotion active, else 0")
    price: float = Field(..., gt=0, description="Our product price")
    competitor_price: float = Field(..., gt=0, description="Competitor product price")
    stockout: int = Field(..., ge=0, le=1, description="1 if out of stock, else 0")


class ForecastResponse(BaseModel):
    predicted_demand: float
    date: str


@app.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest) -> ForecastResponse:
    """Predict daily demand given structured inputs."""
    try:
        prediction = predict_one(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ForecastResponse(predicted_demand=round(prediction, 2), date=payload.date)


# ── Chat endpoint (natural language) ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="Natural language question about demand")


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """
    Accept a plain-English question, parse it, run forecasting inference, return a
    natural-language explanation of the prediction.
    """
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    reply = process_message(payload.message)
    return ChatResponse(reply=reply)
