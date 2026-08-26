import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("crack_model.h5")

# Path to test image
img_path = "test.jpg"   # change this to your image name

# Load and preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Crack Detected")
else:
    print("No Crack Detected")