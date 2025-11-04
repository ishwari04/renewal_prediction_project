# ===============================================================
# app/app.py — Streamlit Dashboard calling FastAPI for predictions
# ===============================================================

import streamlit as st
import pandas as pd
import requests
import io

# ===============================================================
# CONFIGURATION
# ===============================================================
FASTAPI_URL = "https://renewal-prediction-project.onrender.com/predict" 

st.set_page_config(page_title="Policy Renewal Prediction Dashboard", layout="wide")
st.title("📊 Policy Renewal Prediction Dashboard")
st.write("Upload a customer dataset — predictions will be served by the deployed FastAPI model.")

# ===============================================================
# STEP 1: File Upload
# ===============================================================
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(df.head())

        # ===============================================================
        # STEP 2: Send to FastAPI for Prediction
        # ===============================================================
        st.info("🚀 Sending data to FastAPI backend for prediction...")

        # Convert dataframe to JSON
        payload = {"data": df.to_dict(orient="records")}

        # POST request to FastAPI
        response = requests.post(FASTAPI_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            preds = result.get("predictions", [])

            # Append predictions to DataFrame
            df["renewal_probability"] = preds

            st.success(f"✅ Predictions generated for {len(preds)} records")
            st.subheader("📈 Prediction Results")
            st.dataframe(df.head())

            # ===============================================================
            # STEP 3: Download Button
            # ===============================================================
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="renewal_predictions.csv",
                mime="text/csv",
            )

        else:
            st.error(f"❌ FastAPI request failed! Status code: {response.status_code}")
            st.text(response.text)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("⬆️ Please upload a CSV file to start predictions.")
