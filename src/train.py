# src/train.py
import os, json, argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.features import clean_and_split
from src.model import build_mlp
from src.metrics import summarize

MODELS_DIR = "models"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="koi", help="comma-separated: koi,k2,toi")
    args = ap.parse_args()
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = clean_and_split(sources=sources)

    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): w for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

    model = build_mlp(input_dim=X_train.shape[1], n_classes=3)

    ckpt_path = os.path.join(MODELS_DIR, "exoplanet_mlp.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=128,
        class_weight=class_weight,
        callbacks=callbacks, verbose=2
    )

    # Evaluate on test
    test_probs = model.predict(X_test)
    y_pred = np.argmax(test_probs, axis=1)
    summarize(y_test, y_pred)

    # Ensure the model file exists even if checkpoint didn't trigger
    model.save(ckpt_path)

    # Save metadata
    meta = {
        "feature_names": feature_names,
        "label_map": {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2},
        "sources": list(sources)
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
