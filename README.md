# 🧘‍♀️ SuryaPose AI – Real-Time Pose Classification & Injury-Aware Feedback System

**SuryaPose AI** is an intelligent yoga posture classification system that recognizes Surya Namaskar poses in real time and provides feedback to prevent potential injuries. Using deep learning and human pose estimation, the system assists users in maintaining proper form during yoga practice.

---

## 🔍 Features

- ✅ **Real-Time Pose Detection** using TensorFlow.js MoveNet to track 17 body keypoints  
- 🧠 **CNN-Based Pose Classification Model** trained on labeled Surya Namaskar pose data using keypoints (x, y, score)  
- ⚠️ **Injury-Aware Feedback** using joint angle analysis and spatial rules to detect improper alignments  
- 🔊 **Audio Alerts** to notify users when a wrong posture is detected  
- 🌐 **Frontend Web Interface** built with HTML, CSS, JavaScript for live camera feed and feedback  
- 🧪 **Flask Backend API** to serve predictions from a trained `.h5` or `.tflite` model  
- 📉 **Lightweight & Fast** – Suitable for deployment on low-resource devices and local systems  

---

## 🛠️ Technologies Used

- **Frontend:** HTML, CSS, JavaScript, TensorFlow.js  
- **Backend:** Flask, TensorFlow, NumPy, Pandas, Pickle  
- **Modeling:** Convolutional Neural Network (CNN), `.h5` and `.tflite` models  
- **Pose Estimation:** MoveNet (BlazePose optional)  
- **Tools:** VS Code, Google Colab, Git, Postman, Chrome DevTools  

---

## 📊 Dataset

- Collected custom dataset of 8+ Surya Namaskar poses  
- Each sample contains 17 keypoints with x, y coordinates and detection score  
- Preprocessed and augmented for robust training  

---

## 🚀 How It Works

1. Frontend captures webcam frames and extracts keypoints using MoveNet  
2. Keypoints are sent to the Flask backend via REST API  
3. Backend processes the input and predicts the yoga pose class  
4. Based on prediction and joint angles, audio alerts are generated if posture is incorrect  

---

## 📽️ Demo

*(Add your video demo link or GIF preview here)*

