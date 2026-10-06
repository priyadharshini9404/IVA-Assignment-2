# 🔬 IVA VisionLab

### Image & Video Analytics Laboratory

IVA VisionLab is a web-based Image and Video Analytics project that demonstrates classical computer vision and deep-learning based face analysis techniques through an interactive interface.

The project provides four major computer vision operators:

- 🔎 Template Matching
- 👤 Viola–Jones Face Detection
- 🧠 DeepFace
- 🧬 FaceNet

---

## 🚀 Features

### 1. Template Matching

Template Matching is a classical image-processing technique used to locate a smaller image or pattern inside a larger image.

**Input:** Main image and template image  
**Output:** Best matching region with a matching score.

### 2. Viola–Jones

Viola–Jones is a classical real-time object detection algorithm commonly used for face detection.

It uses Haar-like features and a cascade classifier to detect faces.

**Input:** Face image  
**Output:** Detected faces highlighted with bounding boxes.

### 3. DeepFace

DeepFace is a deep-learning based face analysis framework.

It can analyze a face and provide information such as:

- Estimated age
- Gender
- Dominant emotion

**Input:** Face image  
**Output:** Face analysis results.

### 4. FaceNet

FaceNet is a deep-learning based face recognition method.

It converts faces into numerical representations called **embeddings** and compares them using cosine similarity.

**Input:** Two face images  
**Output:** Similarity score and same/different person result.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- DeepFace
- TensorFlow / Keras
- FaceNet
- PyTorch
- Pillow

---

## 📁 Project Structure

```text
IVA-Assignment-2/
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md