# src/features.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from src.datasets import load_all

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

NUMERIC_FEATURES = ["period","duration","depth","prad","impact","snr","steff","slogg","srad"]
CAT_FEATURES = ["mission"]  # optional one-hot
LABEL_MAP = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}

def _prepare_features(df: pd.DataFrame):
    # median-impute numeric features
    for c in NUMERIC_FEATURES:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)

    X_num = df[[c for c in NUMERIC_FEATURES if c in df.columns]].values
    feature_names = [c for c in NUMERIC_FEATURES if c in df.columns]

    ohe = None
    if "mission" in df.columns:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = ohe.fit_transform(df[["mission"]])
        cat_names = [f"mission_{c}" for c in ohe.categories_[0]]
        X = np.hstack([X_num, X_cat])
        feature_names += cat_names
    else:
        X = X_num

    return X, feature_names, ohe

def clean_and_split(sources=("koi",), test_size=0.2, val_size=0.1, random_state=42):
    df = load_all(sources=sources)
    df["label"] = df["disposition"].map(LABEL_MAP).astype(int)

    X, feature_names, ohe = _prepare_features(df)
    y = df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size+val_size), stratify=y, random_state=random_state
    )
    rel_val = val_size / (test_size+val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-rel_val, stratify=y_temp, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train_s)
    np.save(os.path.join(PROCESSED_DIR, "X_val.npy"),   X_val_s)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test_s)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump({"ohe": ohe}, os.path.join(MODELS_DIR, "encoders.joblib"))

    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), feature_names
