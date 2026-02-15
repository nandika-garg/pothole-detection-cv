# Pothole Detection using Computer Vision

## Overview
This project aims to detect potholes on road surfaces using computer vision and deep learning. A Convolutional Neural Network (CNN) model is trained on images of potholes and plain roads to classify whether a pothole is present or not.

---

## Objective
- Detect potholes from road images
- Improve road safety using AI
- Build a basic deep learning model

---

## Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Scikit-learn

---

## Dataset
The dataset contains images divided into two categories:
- Pothole
- Plain road

Dataset is structured as:
dataset/
 ├── train/
 │    ├── pothole/
 │    └── plain/

---

## Methodology
1. Collect images of potholes and normal roads
2. Convert images to grayscale
3. Resize images to fixed size (100x100)
4. Train CNN model using labeled data
5. Save model as .h5 file
6. Use model for prediction

---

## Model Architecture
- Convolutional Layers (feature extraction)
- Activation (ReLU)
- Global Average Pooling
- Fully Connected Layers
- Softmax output (classification)

---

## How to Run

### 1. Train Model
python3 src/main.py
### 2. Predict Image
python3 src/predictor.py

---

## Output
The model predicts:
- Pothole
- Plain Road

Example:
Prediction: Pothole

---

## Project Structure
pothole detection using computer vision/
│
├── dataset/
├── model/
│ └── pothole_model.h5
│
├── src/
│ ├── main.py
│ └── predictor.py
│
├── README.md
├── requirements.txt

---

## Limitations
- Only classifies image, does not detect location
- Limited dataset
- Accuracy can be improved

---

## Future Work
- Real-time pothole detection using video

---

## Conclusion
This project demonstrates how computer vision and deep learning can be used to detect potholes and improve road maintenance systems.

---
