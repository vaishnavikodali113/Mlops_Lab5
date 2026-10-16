import os
import yaml
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Load hyperparameters from params.yaml
p = yaml.safe_load(open("params.yaml"))["train"]
EPOCHS = p["epochs"]
BATCH_SIZE = p["batch_size"]

# Load preprocessed data
d = np.load("data/processed/cifar10.npz")
x_train = d["x_train"] / 255.0
y_train = d["y_train"]

# Build CNN model
model = keras.Sequential([
    layers.Input((32, 32, 3)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
)

os.makedirs("models", exist_ok=True)
model.save("models/model.keras")
print("Model saved to models/model.keras")
