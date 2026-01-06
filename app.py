import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "dog_classifier_mobilenetv2.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.npy")
IMG_SIZE = (224,224)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_PATH, allow_pickle=True)

def smart_resize(img):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim)//2
    top = (h - min_dim)//2
    img = img.crop((left, top, left+min_dim, top+min_dim))
    return img.resize(IMG_SIZE)

def predict_gui(img):
    img = img.convert("RGB")
    img = smart_resize(img)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top5 = preds.argsort()[-5:][::-1]

    return {class_names[i]: float(preds[i]) for i in top5}

iface = gr.Interface(
    fn=predict_gui,
    inputs=gr.Image(type="pil", label="Upload Gambar Anjing"),
    outputs=gr.Label(num_top_classes=5, label="Hasil Prediksi"),
    title="Klasifikasi Ras Anjing",
    description="MobileNetV2 (Fast & Akurat)"
)

iface.launch()
