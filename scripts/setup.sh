#!/usr/bin/env bash
# Full setup sequence — every step confirmed working on kind.
# Run section by section, not all at once, so you can verify each one.
set -e

echo "=== Step 1 — Create kind cluster with fixed port mappings ==="
kind create cluster --name house-price --config=kind-cluster-config.yaml
kubectl wait --for=condition=Ready node --all --timeout=120s
docker ps | grep house-price-control-plane

echo "=== Step 2 — Deploy local registry (localhost:15000) ==="
kubectl apply -f k8s/registry.yaml
kubectl wait --namespace kube-system --for=condition=ready pod --selector=app=registry --timeout=60s
curl -sf http://localhost:15000/v2/ && echo "  Registry OK"

echo "=== Step 3 — Deploy MinIO (localhost:19000 / 19001) ==="
kubectl apply -f k8s/minio.yaml
kubectl wait --namespace minio --for=condition=ready pod --selector=app=minio --timeout=90s
kubectl logs -n minio job/minio-setup

echo "=== Step 4 — Deploy MLflow ==="
kubectl apply -f k8s/mlflow.yaml
kubectl wait --namespace house-price --for=condition=ready pod --selector=app=mlflow --timeout=90s

echo "=== Step 5 — Install Nginx Ingress (kind-specific manifest) ==="
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=120s
kubectl apply -f k8s/ingress-controller-patch.yaml

echo "=== Step 6 — Install Prometheus + Grafana ==="
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values helm/monitoring-values.yaml \
  --wait --timeout 5m
kubectl apply -f k8s/monitoring.yaml

echo "=== Step 7 — Install ArgoCD ==="
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --namespace argocd --for=condition=ready pod --selector=app.kubernetes.io/name=argocd-server --timeout=180s
kubectl patch configmap argocd-cmd-params-cm -n argocd --type merge -p '{"data":{"server.insecure":"true"}}'
kubectl rollout restart deployment argocd-server -n argocd
kubectl rollout status deployment argocd-server -n argocd --timeout=120s
kubectl apply -f k8s/argocd-ingress.yaml

echo ""
echo "=== Base infrastructure is up ==="
echo "  Registry  : http://localhost:15000/v2/"
echo "  MinIO     : http://localhost:19000/minio/health/live"
echo "  Grafana   : http://localhost:18080/grafana  (admin/admin)"
echo "  ArgoCD    : http://localhost:18080/argocd"
echo ""
echo "ArgoCD password:"
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""
echo "Next: build trainer/api/streamlit images, run generate-data-job and train-job"
