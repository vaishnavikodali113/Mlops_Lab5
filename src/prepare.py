import os
import numpy as np
from tensorflow.keras.datasets import cifar10

os.makedirs("data/processed", exist_ok=True)

(x_tr, y_tr), (x_te, y_te) = cifar10.load_data()

np.savez_compressed(
    "data/processed/cifar10.npz",
    x_train=x_tr,
    y_train=y_tr,
    x_test=x_te,
    y_test=y_te,
)

print(f"Saved train: {x_tr.shape}, test: {x_te.shape}")
