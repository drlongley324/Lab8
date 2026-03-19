import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
FEATURES = ["CALC_ACRES", "YEARBUILT", "SIZEAREA"]
TARGET = "APPRAISED_VALUE"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SPLIT = 0.20
EPOCHS = 50
BATCH_SIZE = 32


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


def find_dataset() -> str:
    """
    Try several likely dataset locations so the script works better in RStudio,
    terminal, and project-based execution.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    candidate_paths = [
        os.path.join(script_dir, "data", "Housing_Hamilton_Compressed.csv.gz"),
        os.path.join(cwd, "data", "Housing_Hamilton_Compressed.csv.gz"),
        os.path.join(cwd, "Housing_Hamilton_Compressed.csv.gz"),
        os.path.join(script_dir, "Housing_Hamilton_Compressed.csv.gz"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    searched = "\n".join(candidate_paths)
    raise FileNotFoundError(
        "Dataset not found.\n\n"
        "The script searched these locations:\n"
        f"{searched}\n\n"
        "Fix options:\n"
        "1. Put Housing_Hamilton_Compressed.csv.gz inside a folder named 'data' in your project\n"
        "2. Or place the file in the same folder as train_model.py\n"
        "3. Or run the script from the correct project directory in RStudio"
    )


def main() -> None:
    set_seeds(RANDOM_STATE)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Current working directory:", os.getcwd())
    print("Script directory:", script_dir)

    data_path = find_dataset()
    print("Dataset found at:", data_path)

    print("Loading dataset...")
    df = pd.read_csv(data_path, compression="gzip", low_memory=False)

    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Keep only required columns
    df = df[FEATURES + [TARGET]].copy()

    # Convert to numeric
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean rows
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df = df[df[TARGET] > 0].copy()
    df = df[df["CALC_ACRES"] > 0].copy()
    df = df[df["SIZEAREA"] > 0].copy()
    df = df[(df["YEARBUILT"] >= 1800) & (df["YEARBUILT"] <= 2026)].copy()

    if df.empty:
        raise ValueError("No usable rows remain after cleaning.")

    print(f"Rows after cleaning: {len(df):,}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Recommended stability improvement: train on log target
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(input_dim=X_train_scaled.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]

    print("Training model...")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    print("Evaluating model...")
    test_loss, test_mae_log = model.evaluate(X_test_scaled, y_test, verbose=0)

    y_pred_log = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test.to_numpy())
    mae_dollars = np.mean(np.abs(y_true - y_pred))

    print(f"Test MAE (log scale): {test_mae_log:.4f}")
    print(f"Test MAE (dollars): ${mae_dollars:,.2f}")

    model_path = os.path.join(artifacts_dir, "housing_model.h5")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    features_path = os.path.join(artifacts_dir, "feature_names.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(FEATURES, features_path)

    print("Artifacts saved successfully:")
    print(model_path)
    print(scaler_path)
    print(features_path)


if __name__ == "__main__":
    main()
