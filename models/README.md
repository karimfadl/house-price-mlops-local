# models/

Trained by the train Job inside Kubernetes.
Saved to MinIO at minio://house-price-models/models/house_price_model.pkl.
The API pod downloads it from MinIO on startup.
This directory is git-ignored.
