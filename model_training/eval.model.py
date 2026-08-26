import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = tf.keras.models.load_model("crack_model.h5")

# Dataset path
dataset_path = "dataset"

img_height = 150
img_width = 150
batch_size = 32

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Normalize
normalization_layer = layers.Rescaling(1./255)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Collect predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

# Print metrics
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred))

print("\n📊 CONFUSION MATRIX:")
print(confusion_matrix(y_true, y_pred))