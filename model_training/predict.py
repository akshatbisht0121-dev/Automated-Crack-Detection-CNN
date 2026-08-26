import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model from backend/
MODEL_PATH = os.path.join(BASE_DIR, "..", "backend", "crack_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Path to test image
img_path = os.path.join(BASE_DIR, "test.jpg")   # change this to your image name

# IMPORTANT: must match the size the model was trained on (train_model.py)
IMG_SIZE = (150, 150)

# Load and preprocess image
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)
confidence = float(prediction[0][0])

if confidence > 0.5:
    print(f"Crack Detected ({confidence:.2f})")
else:
    print(f"No Crack Detected ({confidence:.2f})")