"""
API tests — run with: pytest tests/ -v
Uses a mock model so no cluster or MinIO is needed.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MINIO_ENDPOINT",   "http://mock:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("MINIO_BUCKET",     "house-price-models")

mock_model = MagicMock()
mock_model.predict.return_value = np.array([285000.0])

with patch("boto3.client"), patch("pickle.load", return_value=mock_model), \
     patch("builtins.open", create=True):
    from fastapi.testclient import TestClient
    import api as api_module
    api_module.model = mock_model
    client = TestClient(api_module.app)

VALID = {
    "sqft_living": 1800, "bedrooms": 3, "bathrooms": 2,
    "house_age": 10, "distance_city": 8.5, "garage": 1, "school_rating": 7.5,
}


class TestHealth:
    def test_health_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_ready_returns_200(self):
        assert client.get("/ready").status_code == 200


class TestPredict:
    def test_valid_house_returns_200(self):
        assert client.post("/predict", json=VALID).status_code == 200

    def test_response_has_required_fields(self):
        data = client.post("/predict", json=VALID).json()
        assert "predicted_price"  in data
        assert "price_range_low"  in data
        assert "price_range_high" in data
        assert "timestamp"        in data

    def test_confidence_range_wraps_prediction(self):
        data = client.post("/predict", json=VALID).json()
        assert data["price_range_low"]  < data["predicted_price"]
        assert data["price_range_high"] > data["predicted_price"]

    def test_rejects_negative_sqft(self):
        assert client.post("/predict", json={**VALID, "sqft_living": -1}).status_code == 422

    def test_rejects_invalid_garage(self):
        assert client.post("/predict", json={**VALID, "garage": 5}).status_code == 422

    def test_rejects_out_of_range_school_rating(self):
        assert client.post("/predict", json={**VALID, "school_rating": 11.0}).status_code == 422


class TestBatch:
    def test_batch_returns_correct_count(self):
        r = client.post("/predict/batch", json={"houses": [VALID, VALID]})
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_batch_rejects_over_100(self):
        assert client.post("/predict/batch", json={"houses": [VALID] * 101}).status_code == 400


class TestMetrics:
    def test_metrics_returns_prometheus_format(self):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "prediction_requests_total" in r.text


class TestDrift:
    def test_drift_report_returns_json(self):
        r = client.get("/drift/report")
        assert r.status_code == 200
        assert "scores" in r.json() or "note" in r.json()
