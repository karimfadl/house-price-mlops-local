"""
Streamlit frontend — two pages:
  Predict      : fill in house details, see predicted price + feature impact
  Drift monitor: live PSI scores per feature

Runs as a Kubernetes Deployment. Talks to FastAPI via cluster DNS.

Environment variables:
  API_URL  http://house-price-api  (set in deployment-streamlit.yaml)
"""

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

page = st.sidebar.radio("Navigate", ["Predict", "Drift monitor"])
st.sidebar.markdown("---")

try:
    r      = requests.get(f"{API_URL}/health", timeout=2)
    status = "healthy" if r.status_code == 200 else "error"
    color  = "green"
except Exception:
    status = "unreachable"
    color  = "red"

st.sidebar.markdown(f"**API:** :{color}[{status}]")
st.sidebar.caption(API_URL)

if page == "Predict":
    st.title("🏠 House price predictor")
    st.caption("Fill in the details and click Predict.")

    col1, col2 = st.columns(2)
    with col1:
        sqft_living   = st.number_input("Living area (sqft)", 100,  20000, 1800, step=50)
        bedrooms      = st.number_input("Bedrooms",           1,    20,    3)
        bathrooms     = st.number_input("Bathrooms",          1,    10,    2)
        house_age     = st.number_input("House age (years)",  0,    150,   10)
    with col2:
        distance_city = st.number_input("Distance to city (km)", 0.1, 100.0, 8.5, step=0.5)
        school_rating = st.slider("School rating (1-10)", 1.0, 10.0, 7.5, step=0.1)
        garage        = st.selectbox("Garage", [1, 0], format_func=lambda x: "Yes" if x else "No")

    st.markdown("---")

    if st.button("Predict price", type="primary", use_container_width=True):
        payload = {
            "sqft_living": sqft_living, "bedrooms": bedrooms,
            "bathrooms": bathrooms,     "house_age": house_age,
            "distance_city": distance_city, "garage": garage,
            "school_rating": school_rating,
        }
        with st.spinner("Calling model..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                resp.raise_for_status()
                data  = resp.json()
                price = data["predicted_price"]
                lo    = data["price_range_low"]
                hi    = data["price_range_high"]

                st.success(f"### Predicted price: ${price:,}")
                st.info(f"Range: ${lo:,} – ${hi:,}  ({data['confidence_note']})")

                st.markdown("**Feature impact:**")
                impacts = {
                    "Living area":   sqft_living   * 120,
                    "School rating": school_rating * 7_000,
                    "Distance":     -distance_city * 3_000,
                    "House age":    -house_age     * 1_500,
                    "Garage":        garage        * 15_000,
                    "Bedrooms":      bedrooms      * 8_000,
                    "Bathrooms":     bathrooms     * 5_000,
                }
                df_i = pd.DataFrame({
                    "Feature":    list(impacts.keys()),
                    "Impact ($)": list(impacts.values()),
                }).sort_values("Impact ($)", ascending=True)
                st.bar_chart(df_i.set_index("Feature"))

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API.")
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.title("📊 Drift monitor")
    st.caption("PSI recalculates every 50 requests.")

    auto = st.toggle("Auto-refresh every 5s", value=False)

    try:
        data   = requests.get(f"{API_URL}/drift/report", timeout=3).json()
        scores = data.get("scores", {})
        window = data.get("window_size", 0)

        st.markdown(f"**Watching last {window} requests**")

        if not scores:
            st.warning(data.get("note", "No data yet — send some predictions first."))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("OK",    sum(1 for v in scores.values() if v["status"] == "OK"))
            c2.metric("Warn",  sum(1 for v in scores.values() if v["status"] == "WARN"))
            c3.metric("Drift", sum(1 for v in scores.values() if v["status"] == "DRIFT"))

            st.markdown("---")
            for feat, info in scores.items():
                psi    = info["psi"]
                status = info["status"]
                color  = {"OK": "green", "WARN": "orange", "DRIFT": "red"}.get(status, "gray")
                ca, cb, cc = st.columns([3, 1, 1])
                ca.markdown(f"**{feat}**")
                cb.progress(min(psi / 0.3, 1.0))
                cc.markdown(f":{color}[{status}]  `{psi}`")

    except Exception as e:
        st.error(f"Cannot reach drift endpoint: {e}")

    if auto:
        time.sleep(5)
        st.rerun()
