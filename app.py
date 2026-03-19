import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf


ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "housing_model.h5")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Run train_model.py first so artifacts are created."
        )
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Scaler file not found. Run train_model.py first so artifacts are created."
        )
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            "Feature names file not found. Run train_model.py first so artifacts are created."
        )
    return joblib.load(FEATURES_PATH)


st.set_page_config(page_title="Housing Value Predictor", layout="centered")

st.title("Hamilton County Housing Value Predictor")
st.caption("Educational use only. Predictions are approximate and should not be used for appraisal or lending decisions.")

st.subheader("Enter property information")

acres = st.number_input(
    "Land area (acres)",
    min_value=0.01,
    max_value=20.0,
    value=0.25,
    step=0.01
)

yearbuilt = st.number_input(
    "Year built",
    min_value=1900,
    max_value=2026,
    value=2000,
    step=1
)

sizearea = st.number_input(
    "Building area (sq ft)",
    min_value=300,
    max_value=10000,
    value=1800,
    step=50
)

if st.button("Predict Appraised Value"):
    try:
        model = load_model()
        scaler = load_scaler()
        features = load_features()

        input_df = pd.DataFrame([{
            "CALC_ACRES": acres,
            "YEARBUILT": yearbuilt,
            "SIZEAREA": sizearea
        }])

        # Keep exact training feature order
        input_df = input_df[features]

        input_scaled = scaler.transform(input_df)

        # Model predicts log(value), so convert back
        pred_log = model.predict(input_scaled, verbose=0)[0][0]
        pred_value = float(np.expm1(pred_log))

        pred_value = max(pred_value, 0.0)

        st.success(f"Estimated appraised value: ${pred_value:,.0f}")

        st.write("### Inputs used")
        st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure you ran train_model.py first and that the artifacts folder contains the saved model files.")
