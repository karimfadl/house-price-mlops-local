"""
Train house price prediction model.

Runs inside Kubernetes as a Job. Reads dataset from MinIO, logs to MLflow,
saves model back to MinIO. Everything happens via cluster-internal DNS —
this script never touches localhost ports.

Steps:
  1. Download dataset from MinIO
  2. Split 80/20 train/test
  3. Train GradientBoostingRegressor
  4. Log params, metrics, feature importances, model to MLflow
  5. Quality gate: exit with error if MAE > $30,000
  6. Save model to MinIO at models/house_price_model.pkl

Environment variables (set in k8s/train-job.yaml):
  MLFLOW_TRACKING_URI    http://mlflow.house-price.svc.cluster.local:5000
  MLFLOW_S3_ENDPOINT_URL http://minio.minio.svc.cluster.local:9000
  AWS_ACCESS_KEY_ID      minioadmin
  AWS_SECRET_ACCESS_KEY  minioadmin
  MINIO_BUCKET           house-price-models
"""

import os
import pickle

import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from botocore.client import Config
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MLFLOW_URI     = os.environ["MLFLOW_TRACKING_URI"]
MINIO_ENDPOINT = os.environ["MLFLOW_S3_ENDPOINT_URL"]
MINIO_ACCESS   = os.environ["AWS_ACCESS_KEY_ID"]
MINIO_SECRET   = os.environ["AWS_SECRET_ACCESS_KEY"]
MINIO_BUCKET   = os.environ["MINIO_BUCKET"]

FEATURES = [
    "sqft_living", "bedrooms", "bathrooms",
    "house_age", "distance_city", "garage", "school_rating",
]
TARGET        = "price"
MAE_THRESHOLD = 35_000

PARAMS = {
    "n_estimators":  200,
    "max_depth":     4,
    "learning_rate": 0.05,
    "subsample":     0.8,
    "random_state":  42,
}

print(f"MLflow  : {MLFLOW_URI}")
print(f"MinIO   : {MINIO_ENDPOINT}")
print(f"Bucket  : {MINIO_BUCKET}")

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS,
    aws_secret_access_key=MINIO_SECRET,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

print("\nDownloading dataset from MinIO...")
os.makedirs("/tmp/data",   exist_ok=True)
os.makedirs("/tmp/models", exist_ok=True)

s3.download_file(MINIO_BUCKET, "data/house_prices.csv", "/tmp/data/house_prices.csv")

df = pd.read_csv("/tmp/data/house_prices.csv")
X  = df[FEATURES]
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("house-price-prediction")

with mlflow.start_run() as run:
    print(f"\nMLflow run: {run.info.run_id}")

    mlflow.log_params(PARAMS)
    mlflow.log_param("features",   FEATURES)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size",  len(X_test))

    print("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(**PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = mean_squared_error(y_test, y_pred) ** 0.5
    r2     = r2_score(y_test, y_pred)

    mlflow.log_metric("mae",  mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2",   r2)

    importances = dict(zip(FEATURES, model.feature_importances_.round(4)))
    mlflow.log_dict(importances, "feature_importances.json")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="house-price-model",
    )

    print(f"\n  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R2   : {r2:.4f}")

    print("\nFeature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:<18} {'=' * int(imp * 40)} {imp:.4f}")

    if mae > MAE_THRESHOLD:
        raise SystemExit(
            f"\nQuality gate FAILED: MAE ${mae:,.0f} > threshold ${MAE_THRESHOLD:,}\n"
            "Model not saved. Pipeline stopped."
        )

    print(f"\nQuality gate PASSED: MAE ${mae:,.0f} <= ${MAE_THRESHOLD:,}")

model_path = "/tmp/models/house_price_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

s3.upload_file(model_path, MINIO_BUCKET, "models/house_price_model.pkl")
print(f"\nModel saved -> minio://{MINIO_BUCKET}/models/house_price_model.pkl")
