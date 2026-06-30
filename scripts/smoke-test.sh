#!/usr/bin/env bash
# End-to-end smoke test — verifies every piece of the stack actually works,
# not just that pods are Running. Run this after full setup + deployment.
set -e

BASE="http://localhost:18080"
PASS=0
FAIL=0

check() {
  local label="$1" url="$2"
  status=$(curl -s -o /dev/null -w "%{http_code}" "$url")
  if [ "$status" == "200" ]; then
    echo "  OK   $label ($url)"
    PASS=$((PASS+1))
  else
    echo "  FAIL $label ($url) -> HTTP $status"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Infrastructure reachability ==="
check "Registry"          "http://localhost:15000/v2/"
check "MinIO S3 API"      "http://localhost:19000/minio/health/live"
check "Ingress / Streamlit" "$BASE/app"
check "Ingress / API docs"  "$BASE/api/docs"
check "Ingress / MLflow"    "$BASE/mlflow"
check "Ingress / Grafana"   "$BASE/grafana/login"
check "Ingress / ArgoCD"    "$BASE/argocd"

echo ""
echo "=== API functional test ==="
RESPONSE=$(curl -s -X POST "$BASE/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"sqft_living":1800,"bedrooms":3,"bathrooms":2,"house_age":10,"distance_city":8.5,"garage":1,"school_rating":7.5}')
PRICE=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['predicted_price'])" 2>/dev/null || echo "ERROR")
if [[ "$PRICE" =~ ^[0-9]+$ ]]; then
  echo "  OK   Prediction returned: \$$PRICE"
  PASS=$((PASS+1))
else
  echo "  FAIL Prediction did not return a valid price. Response: $RESPONSE"
  FAIL=$((FAIL+1))
fi

echo ""
echo "=== Drift report — sending 35 requests to populate the window ==="
for i in $(seq 1 35); do
  curl -s -X POST "$BASE/api/predict" \
    -H "Content-Type: application/json" \
    -d '{"sqft_living":1800,"bedrooms":3,"bathrooms":2,"house_age":10,"distance_city":8.5,"garage":1,"school_rating":7.5}' \
    > /dev/null
done
DRIFT=$(curl -s "$BASE/api/drift/report")
WINDOW=$(echo "$DRIFT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('window_size',0))" 2>/dev/null || echo "0")
if [ "$WINDOW" -ge 30 ]; then
  echo "  OK   Drift window populated: $WINDOW requests"
  PASS=$((PASS+1))
else
  echo "  FAIL Drift window did not populate. Response: $DRIFT"
  FAIL=$((FAIL+1))
fi

echo ""
echo "=== Prometheus metrics exposed ==="
METRICS=$(curl -s "$BASE/api/metrics")
if echo "$METRICS" | grep -q "prediction_requests_total"; then
  echo "  OK   Prometheus metrics present"
  PASS=$((PASS+1))
else
  echo "  FAIL prediction_requests_total not found in /metrics"
  FAIL=$((FAIL+1))
fi

echo ""
echo "════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
echo "════════════════════════════════════════"
