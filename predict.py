import cv2
import numpy as np
import tensorflow as tf
import pickle
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "fruit_disease_model.h5")
ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")

# Load model and label encoder
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
            "confidence": float(np.max(prediction))
        }
    except Exception as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    test_path = input("🔍 Enter image path to predict: ")
    result = predict_image(test_path)
    print(result)



