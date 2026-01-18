from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

# =========================
# APP CONFIG
# =========================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dog_classifier_mobilenetv2.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.npy")

IMG_SIZE = (224, 224)

# =========================
# GLOBAL MODEL (LAZY LOAD)
# =========================
model = None
class_names = None

def load_model_once():
    """Load model & class names hanya sekali (saat pertama request)"""
    global model, class_names

    if model is None or class_names is None:
        print("🔄 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = np.load(CLASS_PATH, allow_pickle=True).tolist()
        print("✅ Model loaded successfully")

# =========================
# IMAGE PREPROCESSING
# =========================
def smart_resize(img):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize(IMG_SIZE)

def preprocess_image(img):
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    load_model_once()

    results = None
    image_base64 = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            try:
                img = Image.open(file).convert("RGB")
                img = smart_resize(img)

                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                input_tensor = preprocess_image(img)
                preds = model.predict(input_tensor, verbose=0)[0]

                top5_idx = preds.argsort()[-5:][::-1]

                results = [
                    {
                        "label": class_names[i],
                        "confidence": round(float(preds[i]) * 100, 2)
                    }
                    for i in top5_idx
                ]

            except Exception as e:
                error = f"Terjadi kesalahan saat memproses gambar: {str(e)}"

    return render_template(
        "index.html",
        results=results,
        image_base64=image_base64,
        error=error
    )

# =========================
# ENTRY POINT (LOCAL ONLY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
