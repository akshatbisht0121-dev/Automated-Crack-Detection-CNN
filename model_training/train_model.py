import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Dataset path (relative to this file's folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset")

# Where the trained model should end up (used directly by backend/main.py)
MODEL_OUT_PATH = os.path.join(BASE_DIR, "..", "backend", "crack_model.h5")

# Image parameters
img_height = 150
img_width = 150
batch_size = 32

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Normalize images (0-255 -> 0-1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Build CNN Model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

print("\n📊 FINAL METRICS:")
print("Training Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])
print("Training Loss:", history.history['loss'][-1])
print("Validation Loss:", history.history['val_loss'][-1])

# Classification report / confusion matrix on validation set
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred))

print("\n📊 CONFUSION MATRIX:")
print(confusion_matrix(y_true, y_pred))

# Save model directly to backend/crack_model.h5 so main.py can find it
os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
model.save(MODEL_OUT_PATH)

print(f"✅ Model saved successfully to {os.path.abspath(MODEL_OUT_PATH)}")