"""
Generate synthetic house price dataset and push it to MinIO.

Runs inside Kubernetes as a Job. Pushes CSV to MinIO so train.py can
download it. Both this Job and train.py run entirely inside the kind
cluster using cluster-internal DNS — never touching localhost ports.

Price formula:
  sqft_living   -> +$120  per sqft
  bedrooms      -> +$8k   per bedroom
  bathrooms     -> +$5k   per bathroom
  house_age     -> -$1.5k per year
  distance_city -> -$3k   per km from city centre
  garage        -> +$15k  if present
  school_rating -> +$7k   per rating point (1-10)
  base          -> +$50k  always
  noise         -> +-10%  realistic variation

Environment variables (set in k8s/generate-data-job.yaml):
  MINIO_ENDPOINT    http://minio.minio.svc.cluster.local:9000
  MINIO_ACCESS_KEY  minioadmin
  MINIO_SECRET_KEY  minioadmin
  MINIO_BUCKET      house-price-models
"""

import os

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config

SEED   = 42
N      = 2000
BUCKET = os.environ["MINIO_BUCKET"]

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["MINIO_ENDPOINT"],
    aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
    aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

np.random.seed(SEED)

sqft_living   = np.random.randint(500,  5000, N)
bedrooms      = np.random.randint(1,    7,    N)
bathrooms     = np.random.randint(1,    5,    N)
house_age     = np.random.randint(0,    60,   N)
distance_city = np.random.uniform(0.5,  30,   N).round(2)
garage        = np.random.randint(0,    2,    N)
school_rating = np.random.uniform(1.0,  10.0, N).round(1)

base_price = (
    sqft_living   * 120
    + bedrooms    * 8_000
    + bathrooms   * 5_000
    - house_age   * 1_500
    - distance_city * 3_000
    + garage      * 15_000
    + school_rating * 7_000
    + 50_000
)
noise = np.random.normal(0, base_price * 0.10)
price = np.clip(base_price + noise, 50_000, 2_000_000).astype(int)

df = pd.DataFrame({
    "sqft_living":   sqft_living,
    "bedrooms":      bedrooms,
    "bathrooms":     bathrooms,
    "house_age":     house_age,
    "distance_city": distance_city,
    "garage":        garage,
    "school_rating": school_rating,
    "price":         price,
})

os.makedirs("/tmp/data", exist_ok=True)
local_path = "/tmp/data/house_prices.csv"
df.to_csv(local_path, index=False)

print(f"Generated {len(df):,} samples")
print(f"  Price range : ${df['price'].min():,} - ${df['price'].max():,}")
print(f"  Mean price  : ${df['price'].mean():,.0f}")

try:
    s3.head_bucket(Bucket=BUCKET)
except Exception:
    s3.create_bucket(Bucket=BUCKET)
    print(f"Created bucket: {BUCKET}")

s3.upload_file(local_path, BUCKET, "data/house_prices.csv")
print(f"Dataset pushed -> minio://{BUCKET}/data/house_prices.csv")
