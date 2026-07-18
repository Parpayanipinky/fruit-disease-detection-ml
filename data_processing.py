




import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Get absolute path to dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset", "Fruit Disease Data")
IMG_SIZE = 100
X, y = [], []

# Traverse dataset
for fruit_type in os.listdir(DATASET_DIR):
    fruit_path = os.path.join(DATASET_DIR, fruit_type)
    if not os.path.isdir(fruit_path):
        continue

    for condition in os.listdir(fruit_path):
        label = f"{condition}_{fruit_type}"
        condition_path = os.path.join(fruit_path, condition)

        for img_name in os.listdir(condition_path):
            img_path = os.path.join(condition_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
            except:
                print(f"Failed to process: {img_path}")

# Convert to NumPy arrays
X = np.array(X, dtype='float32') / 255.0
y = np.array(y)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Save data
OUT_DIR = os.path.join(BASE_DIR, "..")
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

# Save encoder
with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

print("✅ Preprocessing complete. Files saved.")


