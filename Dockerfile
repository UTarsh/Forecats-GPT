# ── Stage 1: build dependencies ──────────────────────────────────────────────
# We use a slim Python image to keep the final image small.
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only requirements first so Docker caches this layer.
# If requirements.txt doesn't change, pip install is skipped on rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy source code and the trained model
COPY src/ ./src/
COPY models/ ./models/

# Don't run as root in production — create a non-root user
RUN useradd --create-home appuser
USER appuser

# Expose the port uvicorn will listen on
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
