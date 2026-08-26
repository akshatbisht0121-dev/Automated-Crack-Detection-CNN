# # 🔍 Crack Detection Using Deep Learning

A deep learning-based computer vision system that automatically detects cracks in images using a Convolutional Neural Network (CNN).

The project uses **TensorFlow/Keras** for crack classification, **OpenCV** for image processing and annotation, and **FastAPI** to expose the trained model through a REST API.

---

## 🚀 Features

- 🧠 CNN-based crack detection
- 🖼️ Image upload and prediction
- 📊 Crack / No Crack classification
- 🎯 Confidence score for predictions
- 🔴 Automatic crack highlighting on output images
- ⚡ FastAPI REST API
- 📚 Swagger API documentation
- 🐍 Python-based implementation
- 🔧 Modular project structure

---

## 🏗️ Project Architecture

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ├── Resize to 150 × 150
     ├── RGB conversion
     └── Pixel normalization
     │
     ▼
CNN Model
     │
     ▼
Prediction
     │
     ├── Crack Detected
     └── No Crack
     │
     ▼
OpenCV Processing
     │
     ├── Add detection label
     └── Highlight detected region
     │
     ▼
Processed Image
