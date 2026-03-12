from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------
# Optional Streamlit import
# ---------------------------------
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    st = None
    STREAMLIT_AVAILABLE = False


# ---------------------------------
# File paths
# ---------------------------------
ARTIFACT_DIR = Path("deploy_artifacts")
MODEL_PATH = ARTIFACT_DIR / "house_value_model.keras"
ARTIFACTS_PATH = ARTIFACT_DIR / "preprocess_artifacts.joblib"


# ---------------------------------
# Resource loading
# ---------------------------------
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}\n"
            "Run train_model.py first."
        )

    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing artifacts file: {ARTIFACTS_PATH}\n"
            "Run train_model.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)

    required_keys = [
        "numeric_features",
        "categorical_features",
        "numeric_medians",
        "numeric_means",
        "numeric_stds",
        "categorical_fill",
        "category_levels",
        "final_feature_columns",
    ]

    for key in required_keys:
        if key not in artifacts:
            raise KeyError(f"Artifacts file is missing required key: {key}")

    return model, artifacts


# ---------------------------------
# Input preprocessing
# ---------------------------------
def transform_input(X: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    X = X.copy()

    numeric_features = artifacts.get("numeric_features", [])
    categorical_features = artifacts.get("categorical_features", [])

    for col in numeric_features + categorical_features:
        if col not in X.columns:
            X[col] = np.nan

    numeric_frames = []
    for col in numeric_features:
        median_val = artifacts["numeric_medians"].get(col, 0.0)
        mean_val = artifacts["numeric_means"].get(col, 0.0)
        std_val = artifacts["numeric_stds"].get(col, 1.0)

        if std_val == 0 or not np.isfinite(std_val):
            std_val = 1.0

        series = pd.to_numeric(X[col], errors="coerce")
        series = series.fillna(median_val)
        series = (series - mean_val) / std_val
        numeric_frames.append(series.astype(float).rename(col))

    X_num = pd.concat(numeric_frames, axis=1) if numeric_frames else pd.DataFrame(index=X.index)

    dummy_frames = []
    for col in categorical_features:
        fill_val = artifacts["categorical_fill"].get(col, "Missing")
        allowed = artifacts["category_levels"].get(col, [fill_val])

        series = X[col].astype("string")
        series = series.fillna(fill_val).replace({pd.NA: fill_val}).astype(str)
        series = series.apply(lambda v: v if v in allowed else fill_val)

        cat_dtype = pd.CategoricalDtype(categories=allowed)
        series = series.astype(cat_dtype)

        dummies = pd.get_dummies(series, prefix=col, dtype=float)
        dummy_frames.append(dummies)

    X_cat = pd.concat(dummy_frames, axis=1) if dummy_frames else pd.DataFrame(index=X.index)

    X_out = pd.concat([X_num, X_cat], axis=1)

    final_cols = artifacts.get("final_feature_columns", [])
    if final_cols:
        X_out = X_out.reindex(columns=final_cols, fill_value=0.0)

    X_out = X_out.fillna(0.0)
    return X_out


# ---------------------------------
# Prediction helper
# ---------------------------------
def predict_value(user_input: dict, model, artifacts) -> float:
    X_input = pd.DataFrame([user_input])
    X_ready = transform_input(X_input, artifacts)
    X_ready_np = X_ready.to_numpy(dtype=np.float32)

    pred = model.predict(X_ready_np, verbose=0).ravel()[0]

    if not np.isfinite(pred):
        raise ValueError("Prediction returned a non-finite number.")

    return float(pred)


# ---------------------------------
# Streamlit UI
# ---------------------------------
def run_streamlit_app():
    st.set_page_config(
        page_title="Hamilton County Housing Value Predictor",
        layout="centered",
    )

    st.title("Hamilton County Housing Value Predictor")
    st.write("Predict `APPRAISED_VALUE` using the trained neural network.")

    try:
        model, artifacts = load_resources()
    except Exception as e:
        st.error(str(e))
        st.stop()

    numeric_features = artifacts.get("numeric_features", [])
    categorical_features = artifacts.get("categorical_features", [])

    st.subheader("Property Inputs")

    user_input = {}

    for col in numeric_features:
        default_val = float(artifacts["numeric_medians"].get(col, 0.0))
        step_val = 0.1 if "ACRES" in col.upper() else 1000.0
        fmt = "%.4f" if "ACRES" in col.upper() else "%.2f"

        user_input[col] = st.number_input(
            label=col,
            value=default_val,
            step=step_val,
            format=fmt,
        )

    for col in categorical_features:
        options = artifacts["category_levels"].get(col, [])
        fill_val = artifacts["categorical_fill"].get(col, "Missing")

        dropdown_options = [""] + options
        default_index = 0
        if fill_val in dropdown_options:
            default_index = dropdown_options.index(fill_val)

        user_input[col] = st.selectbox(
            label=col,
            options=dropdown_options,
            index=default_index,
        )

    if st.button("Predict appraised value"):
        try:
            prediction = predict_value(user_input, model, artifacts)
            st.success(f"Predicted APPRAISED_VALUE: ${prediction:,.0f}")

            with st.expander("Show input values"):
                st.dataframe(pd.DataFrame([user_input]))

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------------------------
# Safe console mode
# ---------------------------------
def run_console_message():
    print("\nThis file is meant to be launched with Streamlit.\n")
    print("Your current error happened because Streamlit is not installed in the")
    print("Python environment that RStudio/reticulate is using.\n")
    print("To run the app correctly, install Streamlit and launch it with:\n")
    print("    streamlit run app.py\n")
    print("or:\n")
    print("    python -m streamlit run app.py\n")
    print("\nIf you are only testing imports inside RStudio, this file now imports safely.")
    print("It will not crash just because Streamlit is missing.\n")


# ---------------------------------
# Main entry point
# ---------------------------------
def main():
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        run_console_message()


if __name__ == "__main__":
    main()
