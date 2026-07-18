import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Load preprocessed data
X_train = np.load(os.path.join(ROOT_DIR, "X_train.npy"))
X_test = np.load(os.path.join(ROOT_DIR, "X_test.npy"))
y_train = np.load(os.path.join(ROOT_DIR, "y_train.npy"))
y_test = np.load(os.path.join(ROOT_DIR, "y_test.npy"))

# Load label encoder
with open(os.path.join(ROOT_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

num_classes = len(encoder.classes_)

# Build model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
base_model.trainable = False  # Freeze feature extractor

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Checkpoint
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

checkpoint_path = os.path.join(MODELS_DIR, "fruit_disease_model.h5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

# Train
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=32,
                    callbacks=[checkpoint])

print("✅ Model training complete. Saved to models/fruit_disease_model.h5")

# Evaluate model
print("\n🔍 Generating Evaluation Metrics...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (weighted): {f1:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(16, 16))  # Larger figure size
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)

disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix", fontsize=18)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "confusion_matrix_readable.png"), dpi=300)
plt.show()

# Accuracy/Loss plot
def plot_training_history(history):
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "training_accuracy.png"), dpi=300)
    plt.show()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "training_loss.png"), dpi=300)
    plt.show()

plot_training_history(history)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')



