import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


MODEL_PATH = "artifacts/housing_model.keras"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"
FEATURE_INFO_PATH = "artifacts/feature_info.json"


def transform_with_preprocessor(X: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    X = X.copy()

    numeric_out = pd.DataFrame(index=X.index)
    for col in preprocessor["numeric_features"]:
        mean_val = preprocessor["numeric_stats"][col]["mean"]
        std_val = preprocessor["numeric_stats"][col]["std"]
        numeric_out[col] = (pd.to_numeric(X[col], errors="coerce") - mean_val) / std_val

    categorical_parts = []
    for col in preprocessor["categorical_features"]:
        levels = preprocessor["categorical_levels"][col]
        cat_series = X[col].astype(str)

        dummies = pd.get_dummies(cat_series, prefix=col)
        expected_cols = [f"{col}_{level}" for level in levels]
        dummies = dummies.reindex(columns=expected_cols, fill_value=0)
        categorical_parts.append(dummies)

    if categorical_parts:
        categorical_out = pd.concat(categorical_parts, axis=1)
        X_out = pd.concat([numeric_out, categorical_out], axis=1)
    else:
        X_out = numeric_out.copy()

    X_out = X_out.fillna(0.0)
    X_out = X_out.reindex(columns=preprocessor["feature_columns"], fill_value=0)

    return X_out


@st.cache_resource
def load_model_and_preprocessor():
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
        feature_info = json.load(f)
    return model, preprocessor, feature_info


st.set_page_config(page_title="Hamilton County Housing Value Predictor", layout="centered")

st.title("Hamilton County Housing Value Predictor")

model, preprocessor, feature_info = load_model_and_preprocessor()
category_options = feature_info["category_options"]

acres = st.number_input("Land area (acres)", min_value=0.01, max_value=20.0, value=0.25, step=0.01)
land_use = st.selectbox("Land use description", options=category_options["LAND_USE_CODE_DESC"])
neighborhood = st.selectbox("Neighborhood description", options=category_options["NEIGHBORHOOD_CODE_DESC"])
zoning = st.selectbox("Zoning description", options=category_options["ZONING_DESC"])
property_type = st.selectbox("Property type description", options=category_options["PROPERTY_TYPE_CODE_DESC"])

if st.button("Predict Appraised Value"):
    input_df = pd.DataFrame({
        "CALC_ACRES": [acres],
        "LAND_USE_CODE_DESC": [land_use],
        "NEIGHBORHOOD_CODE_DESC": [neighborhood],
        "ZONING_DESC": [zoning],
        "PROPERTY_TYPE_CODE_DESC": [property_type],
    })

    X_input = transform_with_preprocessor(input_df, preprocessor)
    X_input = X_input.astype(np.float32).to_numpy()

    pred_log = model.predict(X_input, verbose=0)[0][0]
    pred_value = np.expm1(pred_log)

    st.success(f"Estimated appraised value: ${pred_value:,.0f}")
