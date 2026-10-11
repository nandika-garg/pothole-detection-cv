

import cv2
import numpy as np
from keras.models import load_model


IMG_SIZE = 100

model = load_model("model/pothole_model.h5")

image_path = "dataset/test/pothole/3.jpg"



img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))


img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

img = img / 255.0


prediction = model.predict(img)

result = np.argmax(prediction)


if result == 1:
    print("Prediction: Pothole")
else:
    print("Prediction: Plain Road")