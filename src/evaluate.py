import json
import numpy as np
from tensorflow import keras

# Load test data
d = np.load("data/processed/cifar10.npz")
x_test = d["x_test"] / 255.0
y_test = d["y_test"]

# Load trained model
model = keras.models.load_model("models/model.keras")

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)

metrics = {"loss": float(loss), "accuracy": float(acc)}
json.dump(metrics, open("metrics.json", "w"), indent=2)

print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
