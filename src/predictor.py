# predictor.py
# Simple pothole detection

import cv2
import numpy as np
from keras.models import load_model

IMG_SIZE = 100

# Load trained model
model = load_model("model/pothole_model.h5")

# Change this image path if needed
image_path = "dataset/train/pothole/1.jpg"

# Read image
img = cv2.imread(image_path, 0)

if img is None:
    print("Error: Image not found")
    exit()

# Preprocess
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
img = img / 255.0

# Predict
prediction = model.predict(img)

result = np.argmax(prediction)

if result == 1:
    print("Prediction: Pothole")
else:
    print("Prediction: Plain Road")
