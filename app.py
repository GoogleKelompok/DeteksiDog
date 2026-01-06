from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dog_classifier_mobilenetv2.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.npy")

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_PATH, allow_pickle=True)

# IMAGE PREPROCESS
def smart_resize(img):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize(IMG_SIZE)

# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    image_base64 = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            # ======================
            # LOAD & PREVIEW IMAGE
            # ======================
            img = Image.open(file).convert("RGB")
            img = smart_resize(img)

            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # ======================
            # MODEL PREDICTION
            # ======================
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)[0]
            top5_idx = preds.argsort()[-5:][::-1]

            results = [
                {
                    "label": class_names[i],
                    "confidence": round(float(preds[i]) * 100, 2)
                }
                for i in top5_idx
            ]

    return render_template(
        "index.html",
        results=results,
        image_base64=image_base64
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

