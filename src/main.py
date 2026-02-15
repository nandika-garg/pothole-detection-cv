# main.py
# Training module for pothole detection (Prototype Version)

import os
print("Current working directory:", os.getcwd())

import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Dropout, GlobalAveragePooling2D
from sklearn.utils import shuffle
from keras.utils import to_categorical

# -------------------------------
# Configuration
# -------------------------------
IMG_SIZE = 100
EPOCHS = 5

print("Loading dataset...")

# -------------------------------
# Load pothole images
# -------------------------------
pothole_images = glob.glob("dataset/train/pothole/*")

# -------------------------------
# Load plain images
# -------------------------------
plain_images = glob.glob("dataset/train/plain/*")

# DEBUG
print("Pothole images found:", len(pothole_images))
print("Plain images found:", len(plain_images))

data = []
labels = []

# -------------------------------
# Process pothole images
# -------------------------------
for path in pothole_images:
    img = cv2.imread(path, 0)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)

# -------------------------------
# Process plain images
# -------------------------------
for path in plain_images:
    img = cv2.imread(path, 0)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)

data = np.array(data)
labels = np.array(labels)

print("Total images:", len(data))

# -------------------------------
# SAFETY CHECK (VERY IMPORTANT)
# -------------------------------
if len(data) == 0:
    print("ERROR: No images found. Check dataset path.")
    exit()

# -------------------------------
# Shuffle dataset
# -------------------------------
data, labels = shuffle(data, labels)

# -------------------------------
# Reshape for CNN
# -------------------------------
data = data.reshape(data.shape[0], IMG_SIZE, IMG_SIZE, 1)

# Normalize
data = data / 255.0

# Convert labels
labels = to_categorical(labels)

# -------------------------------
# Build model
# -------------------------------
print("Building model...")

model = Sequential()

model.add(Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Conv2D(32, (3,3), activation='relu'))

model.add(GlobalAveragePooling2D())

model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

# -------------------------------
# Compile
# -------------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model compiled successfully")

# -------------------------------
# Train
# -------------------------------
print("Training model...")

history = model.fit(data, labels, epochs=EPOCHS, validation_split=0.1)

# -------------------------------
# Save model
# -------------------------------
print("Saving model...")

model.save("model/pothole_model.h5")

print("Model saved successfully (Prototype)")

