# src/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_mlp(input_dim: int, n_classes: int = 3, hidden=[128, 64, 32], dropout=0.2):
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for h in hidden:
        x = layers.Dense(h, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs, outputs, name="exoplanet_mlp")
    model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)
    return model
