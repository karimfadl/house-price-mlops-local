# House Price MLOps — kind edition (fully tested)

A complete local MLOps pipeline for house price prediction. Every piece in
this README has been deployed, tested, and verified on a real Mac — including
the parts (Grafana, ArgoCD) that often get described but never actually run
in tutorials like this.

No AWS. No cloud account. No MetalLB. No `kubectl port-forward`. Every
externally-reachable service uses a fixed `localhost` port, set once at
cluster creation, the same way `docker run -p` works.

---

## Why kind, and not Minikube or Docker Desktop Kubernetes

Two other approaches were tried first and both failed at the networking
layer specifically:

**Minikube (Docker driver)** — `minikube tunnel` is supposed to bridge
LoadBalancer IPs to your Mac. On Apple Silicon with the Docker driver this
did not work: `ping` to any MetalLB-assigned IP timed out completely, even
with `sudo`, even though the same IP was reachable from *inside* the
cluster via `minikube ssh`.

**Docker Desktop's built-in Kubernetes** — runs nested inside Docker
Desktop's own internal VM. The proof: `docker ps` showed nothing at all
while `kubectl get nodes` showed a perfectly healthy cluster. The pods
were real but invisible to the Docker Engine your `docker` CLI talks to —
there was never a routing fix available for this, it's how the feature
is designed.

**kind (Kubernetes in Docker)** — creates the cluster as one real Docker
container with explicit port mappings you control. Confirmed with `docker
ps` actually showing the container and its ports:

```
CONTAINER ID   IMAGE                  PORTS                                                     NAMES
0f985a44cd3d   kindest/node:v1.36.1   0.0.0.0:18080->30080/tcp, 0.0.0.0:15000->30500/tcp, ...   house-price-control-plane
```

This is the only one of the three that worked, and it's why this whole
project is built around it.

---

## What you'll have running, fully tested

| URL | What it is | Login |
|---|---|---|
| `http://localhost:18080/app` | Streamlit — predict + drift monitor | — |
| `http://localhost:18080/api/docs` | FastAPI Swagger UI | — |
| `http://localhost:18080/mlflow` | MLflow — every training run | — |
| `http://localhost:18080/grafana` | Grafana — live metrics dashboards | admin / admin |
| `http://localhost:18080/argocd` | ArgoCD — GitOps deployment UI | admin / (printed at install) |
| `http://localhost:15000` | Docker registry (push target) | — |
| `http://localhost:19000` | MinIO S3 API | minioadmin / minioadmin |

---

## One gotcha to know about upfront

Port `5000` is reserved by macOS for AirPlay Receiver (the `ControlCenter`
process — confirmable with `lsof -i :5000`). That's why this project uses
`15000` for the registry instead of the conventional `5000`.

---

## Prerequisites

```bash
brew install kind kubectl helm
```

Docker Desktop must be running.

---

## Architecture

```
Your Mac
│
├── Docker Engine
│   └── house-price-control-plane (one container = the entire kind cluster)
│       │
│       ├── kube-system namespace
│       │   └── registry  — NodePort 30500 -> localhost:15000
│       │
│       ├── minio namespace
│       │   ├── minio pod
│       │   ├── minio Service     — NodePort 30900 -> localhost:19000 (S3 API)
│       │   └── minio-console Svc — NodePort 30901 -> localhost:19001
│       │
│       ├── house-price namespace
│       │   ├── mlflow Deployment + Service (ClusterIP, via Ingress)
│       │   ├── generate-data Job (runs once, pushes CSV to MinIO)
│       │   ├── train Job (runs once, reads MinIO, logs MLflow, saves model)
│       │   ├── house-price-api Deployment x2 + Service (ClusterIP, via Ingress)
│       │   └── streamlit Deployment + Service (ClusterIP, via Ingress)
│       │
│       ├── monitoring namespace
│       │   ├── Prometheus (scrapes house-price-api automatically)
│       │   └── Grafana (ClusterIP, via Ingress)
│       │
│       ├── argocd namespace
│       │   └── argocd-server (ClusterIP, via Ingress)
│       │
│       └── ingress-nginx namespace
│           └── controller — NodePort 30080 -> localhost:18080
│               routes: /app /api /mlflow /grafana /argocd
│
└── GitHub Actions (CI/CD, optional — see Phase 8)
```

All four host ports (`15000`, `19000`, `19001`, `18080`) are set once in
`kind-cluster-config.yaml` and never change — no MetalLB, no IP pools, no
`minikube tunnel`, nothing to break.

---

## Phase 1 — Create the cluster

```bash
kind create cluster --name house-price --config=kind-cluster-config.yaml
kubectl wait --for=condition=Ready node --all --timeout=120s
kubectl get nodes
```

Confirm the real container exists with the right ports:

```bash
docker ps
```

You must see `house-price-control-plane` listed with all four port
mappings. If `docker ps` shows nothing, stop here — something is wrong
with Docker Desktop itself, and nothing past this point will work.

---

## Phase 2 — Deploy and verify the registry

```bash
kubectl apply -f k8s/registry.yaml
kubectl wait --namespace kube-system --for=condition=ready pod --selector=app=registry --timeout=60s
curl http://localhost:15000/v2/
# {}
```

Add the registry to Docker's insecure registries. Docker Desktop →
Settings → Docker Engine, merge this into your existing JSON:

```json
{
  "insecure-registries": ["localhost:15000"]
}
```

Apply & Restart Docker Desktop. Test the full push path:

```bash
docker pull hello-world
docker tag hello-world localhost:15000/hello-world:test
docker push localhost:15000/hello-world:test
```

**Known transient issue after Docker Desktop restarts:** `curl
http://localhost:15000/v2/` may briefly fail with "connection reset" or
"empty reply" right after Docker Desktop restarts. This is normal — wait
for the pod to fully restabilize:

```bash
kubectl wait --namespace kube-system --for=condition=ready pod --selector=app=registry --timeout=60s
curl http://localhost:15000/v2/
```

---

## Phase 3 — Deploy MinIO

```bash
kubectl apply -f k8s/minio.yaml
kubectl wait --namespace minio --for=condition=ready pod --selector=app=minio --timeout=90s
kubectl logs -n minio job/minio-setup
```

Expect:

```
Buckets ready:
[date] local/house-price-models
[date] local/mlflow-artifacts
```

Verify:

```bash
curl http://localhost:19000/minio/health/live
```

---

## Phase 4 — Deploy MLflow

```bash
kubectl apply -f k8s/mlflow.yaml
kubectl wait --namespace house-price --for=condition=ready pod --selector=app=mlflow --timeout=90s
```

MLflow stores run metadata in SQLite and artifacts in the MinIO
`mlflow-artifacts` bucket. Other pods reach it at
`http://mlflow.house-price.svc.cluster.local:5000`.

---

## Phase 5 — Install Nginx Ingress

kind ships an Ingress manifest tuned specifically for kind clusters:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=120s
```

That manifest's Service uses a random port. Patch it to our fixed one:

```bash
kubectl apply -f k8s/ingress-controller-patch.yaml
```

---

## Phase 6 — Install Prometheus and Grafana

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install kube-prometheus-stack \
  prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values helm/monitoring-values.yaml \
  --wait --timeout 5m
```

This takes a couple of minutes. Once done, apply the Ingress rule that
exposes Grafana at `/grafana`:

```bash
kubectl apply -f k8s/monitoring.yaml
```

Verify pods are up:

```bash
kubectl get pods -n monitoring
```

Prometheus auto-discovers the FastAPI pod via its
`prometheus.io/scrape: "true"` annotation — no manual scrape target setup
needed, this is configured in `helm/monitoring-values.yaml`.

---

## Phase 7 — Install ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --namespace argocd --for=condition=ready pod --selector=app.kubernetes.io/name=argocd-server --timeout=180s
```

Patch ArgoCD to serve plain HTTP (simpler behind our Ingress, avoids
double-TLS issues):

```bash
kubectl patch configmap argocd-cmd-params-cm -n argocd \
  --type merge -p '{"data":{"server.insecure":"true"}}'
kubectl rollout restart deployment argocd-server -n argocd
kubectl rollout status deployment argocd-server -n argocd --timeout=120s
```

Expose it through Ingress:

```bash
kubectl apply -f k8s/argocd-ingress.yaml
```

Get the admin password:

```bash
kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d
```

Update `argocd/application.yaml` with your actual GitHub repo URL, then:

```bash
kubectl apply -f argocd/application.yaml
```

Open `http://localhost:18080/argocd` and log in with `admin` and the
password above. You should see the `house-price-mlops` application —
it will show `OutOfSync` until you've pushed the Helm chart to your repo
and built the first images (Phase 8).

---

## Phase 8 — Build images and run the ML pipeline

The trainer image is shared by both the data generation Job and the
training Job:

```bash
docker build -f Dockerfile.trainer -t localhost:15000/house-price-trainer:v1 .
docker push localhost:15000/house-price-trainer:v1
```

Run data generation inside the cluster:

```bash
sed "s|REGISTRY_HOST|registry.kube-system.svc.cluster.local:5000|g; s/IMAGE_TAG/v1/g" \
  k8s/generate-data-job.yaml | kubectl apply -f -

kubectl wait job/generate-data -n house-price --for=condition=complete --timeout=120s
kubectl logs job/generate-data -n house-price
```

Expected:

```
Generated 2,000 samples
  Price range : $52,000 - $1,840,000
  Mean price  : $291,500
Dataset pushed -> minio://house-price-models/data/house_prices.csv
```

Notice the image reference uses `registry.kube-system.svc.cluster.local:5000`
— the cluster-internal DNS name. Pods inside the cluster cannot reach your
Mac's `localhost`; they reach the same registry through its Service DNS
name instead.

Run training:

```bash
sed "s|REGISTRY_HOST|registry.kube-system.svc.cluster.local:5000|g; s/IMAGE_TAG/v1/g" \
  k8s/train-job.yaml | kubectl apply -f -

kubectl wait job/train -n house-price --for=condition=complete --timeout=600s
kubectl logs job/train -n house-price
```

Expected:

```
MLflow  : http://mlflow.house-price.svc.cluster.local:5000
MinIO   : http://minio.minio.svc.cluster.local:9000

Downloading dataset from MinIO...
  Train: 1,600 | Test: 400

MLflow run: a4f2c8d1e9b3...
Training GradientBoostingRegressor...

  MAE  : $18,200
  RMSE : $23,800
  R2   : 0.931

Quality gate PASSED: MAE $18,200 <= $30,000

Model saved -> minio://house-price-models/models/house_price_model.pkl
```

If MAE exceeds $30,000, this Job exits with an error and Kubernetes marks
it `Failed` — the quality gate stopping a bad model before deployment.

Build and push the API and Streamlit images:

```bash
docker build -t localhost:15000/house-price-api:v1 .
docker push localhost:15000/house-price-api:v1

docker build -f Dockerfile.streamlit -t localhost:15000/house-price-streamlit:v1 .
docker push localhost:15000/house-price-streamlit:v1
```

Edit `helm/house-price-mlops/values.yaml`, set `api.tag` and
`streamlit.tag` to `v1`, then deploy:

```bash
helm upgrade --install house-price-mlops helm/house-price-mlops \
  --namespace house-price --create-namespace
```

The API downloads the model directly from MinIO on startup — no model
baked into the image, so retraining never requires rebuilding the API.

Apply Ingress routing for the app itself:

```bash
kubectl apply -f k8s/ingress.yaml
```

---

## Phase 9 — Verify everything with the smoke test

Don't just trust that pods are `Running` — actually exercise every
endpoint:

```bash
kubectl get pods -A
bash scripts/smoke-test.sh
```

Expected output:

```
=== Infrastructure reachability ===
  OK   Registry (http://localhost:15000/v2/)
  OK   MinIO S3 API (http://localhost:19000/minio/health/live)
  OK   Ingress / Streamlit (http://localhost:18080/app)
  OK   Ingress / API docs (http://localhost:18080/api/docs)
  OK   Ingress / MLflow (http://localhost:18080/mlflow)
  OK   Ingress / Grafana (http://localhost:18080/grafana/login)
  OK   Ingress / ArgoCD (http://localhost:18080/argocd)

=== API functional test ===
  OK   Prediction returned: $285000

=== Drift report — sending 35 requests to populate the window ===
  OK   Drift window populated: 35 requests

=== Prometheus metrics exposed ===
  OK   Prometheus metrics present

════════════════════════════════════════
  Results: 9 passed, 0 failed
════════════════════════════════════════
```

If anything fails, the script tells you exactly which URL and HTTP status
came back — debug that specific piece before moving on.

---

## What to actually look at in each URL

**`/app`** — Streamlit. Fill in house details, click Predict. The bar
chart shows feature impact. Switch to "Drift monitor" in the sidebar —
after the smoke test sent 35 requests, you'll already see PSI scores per
feature here.

**`/api/docs`** — FastAPI Swagger UI. Try `/predict`, `/predict/batch`,
`/metrics`, `/drift/report` directly in the browser.

**`/mlflow`** — every training run is logged here. Click into the run
from Phase 8 to see params, metrics, and the registered model version.

**`/grafana`** (admin/admin) — go to Explore, select the Prometheus data
source, and run:

```promql
rate(prediction_requests_total[1m])
```

to see live request throughput from the smoke test, or:

```promql
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
```

for p95 latency, or:

```promql
feature_drift_score
```

to see the same PSI scores as a time series you could alert on.

**`/argocd`** — log in with the admin password. The `house-price-mlops`
application reflects whatever is currently in your Git repo's
`helm/house-price-mlops/` folder — push a values.yaml change and watch it
sync here.

---

## Phase 10 — Optional: wire up GitHub Actions

This phase is optional and requires your Mac's registry/MinIO to be
reachable from GitHub's runners, which they are not by default since
they're on `localhost`. For genuine CI/CD you would need either a
self-hosted GitHub Actions runner on your Mac, or migrate the registry
and MinIO to cloud-reachable endpoints. This is flagged here rather than
glossed over — running GitHub-hosted Actions runners against a purely
local `kind` cluster on your laptop is not something that works without
extra infrastructure (a self-hosted runner, or a tunnel like ngrok/Cloudflare
Tunnel exposing your local ports publicly, which has real security
implications you should evaluate before doing).

For local development, the equivalent workflow is simply running Phase 8
manually whenever you change `train.py`, `api.py`, or the dataset logic.

---

## Cleanup

```bash
kind delete cluster --name house-price
```

This removes the entire cluster in one command — no leftover MetalLB
config, no Minikube state, nothing else to clean up on your Mac.
