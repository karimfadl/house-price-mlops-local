FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/build/pkg \
    fastapi uvicorn pydantic scikit-learn \
    prometheus-client numpy boto3

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /build/pkg /usr/local
COPY api.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
