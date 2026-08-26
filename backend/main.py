from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import cv2
import os

app = FastAPI()

# Path to the model file, relative to this file's location.
# This makes the app work no matter which directory you launch uvicorn from.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crack_model.h5")

IMG_SIZE = (150, 150)  # must match the size used in train_model.py

# Load model once at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# Home route (to avoid 404 on /)
@app.get("/")
def home():
    return {"message": "Crack Detection API is running 🚀"}


# Prediction route
@app.post("/predict/", response_class=StreamingResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model is not loaded. Check server logs / MODEL_PATH."}

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img = np.array(image)

        # Resize for model
        img_resized = cv2.resize(img, IMG_SIZE) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        # Prediction
        prediction = model.predict(img_input)
        confidence = float(prediction[0][0])

        # Label
        if confidence > 0.5:
            label = f"Crack Detected ({confidence:.2f})"
            color = (255, 0, 0)
        else:
            label = f"No Crack ({confidence:.2f})"
            color = (0, 255, 0)

        # Edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw boxes
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 20 and h > 20:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Overlay label text
        cv2.putText(
            img,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

        # Convert to image
        result_image = Image.fromarray(img)

        buf = io.BytesIO()
        result_image.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=result.jpg"}
        )

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)