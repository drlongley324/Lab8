from pathlib import Path
import joblib

OUT_DIR = Path("deploy_artifacts")
OUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUT_DIR / "house_value_model.h5"
PREP_PATH = OUT_DIR / "preprocess.joblib"

model.save(MODEL_PATH)
joblib.dump(preprocess, PREP_PATH)

print("Saved model to:", MODEL_PATH)
print("Saved preprocess pipeline to:", PREP_PATH)
