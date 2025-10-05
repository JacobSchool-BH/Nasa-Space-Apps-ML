# src/predict.py
import os, json, argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

MODELS_DIR = "models"

def load_artifacts(model_path, scaler_path, enc_path, meta_path):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    enc = joblib.load(enc_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, scaler, enc, meta

def _prepare_input(df: pd.DataFrame, feature_names: list[str], encoders):
    # Expected numeric features (unified)
    num_cols = ["period","duration","depth","prad","impact","snr","steff","slogg","srad"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    X_num = df[[c for c in num_cols if c in df.columns]].values
    X = X_num

    # Handle mission one-hot if present in training
    ohe = encoders.get("ohe", None) if isinstance(encoders, dict) else encoders
    if ohe is not None:
        if "mission" not in df.columns:
            df["mission"] = "kepler"
        X_cat = ohe.transform(df[["mission"]])
        X = np.hstack([X_num, X_cat])

    return X

def predict_csv(in_csv: str, out_csv: str, model_path: str, scaler_path: str, enc_path: str, meta_path: str):
    model, scaler, enc, meta = load_artifacts(model_path, scaler_path, enc_path, meta_path)
    feature_names = meta["feature_names"]

    df = pd.read_csv(in_csv)
    X = _prepare_input(df, feature_names, enc)
    Xs = scaler.transform(X)

    probs = model.predict(Xs)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    rev = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
    df["prediction"] = [rev[int(i)] for i in pred]
    df["confidence"] = conf
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions → {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", default="predictions.csv")
    ap.add_argument("--model_path", default=os.path.join(MODELS_DIR, "exoplanet_mlp.keras"))
    ap.add_argument("--scaler_path", default=os.path.join(MODELS_DIR, "scaler.joblib"))
    ap.add_argument("--enc_path", default=os.path.join(MODELS_DIR, "encoders.joblib"))
    ap.add_argument("--meta_path", default=os.path.join(MODELS_DIR, "meta.json"))
    args = ap.parse_args()
    predict_csv(args.in_csv, args.out_csv, args.model_path, args.scaler_path, args.enc_path, args.meta_path)
