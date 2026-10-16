import io
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
from tensorflow import keras

app = Flask(__name__, template_folder="../templates")

# Load model once at startup
model = keras.models.load_model("models/model.keras")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file uploaded"), 400

    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((32, 32))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        # Return top-3 predictions for richer UI
        top3 = sorted(
            [{"label": CLASSES[i], "confidence": round(float(probs[i]), 3)}
             for i in range(10)],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3]
        return jsonify(label=CLASSES[idx], confidence=round(float(probs[idx]), 3), top3=top3)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
