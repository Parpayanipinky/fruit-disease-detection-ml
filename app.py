from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle

# Init Flask
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and label encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "fruit_disease_model.h5")
ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")

model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (100, 100))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0]
        class_index = np.argmax(prediction)
        class_label = encoder.inverse_transform([class_index])[0]
        disease, fruit = class_label.split("_", 1)

        return {
            "fruit": fruit,
            "disease": disease,
            "confidence": f"{np.max(prediction)*100:.2f}%"
        }
    except Exception as e:
        return {"error": str(e)}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            result = predict_image(filepath)
            result["image"] = file.filename
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
