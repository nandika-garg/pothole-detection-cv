# Pothole Detection using Computer Vision

## Project Description
This project detects potholes on road images using a Convolutional Neural Network (CNN). It classifies road images into two categories — **pothole** and **plain road** — as a step toward automated road-condition monitoring.

## Features
* Binary image classification using a CNN built with TensorFlow/Keras
* Detects potholes from static road images
* Lightweight architecture suitable for quick training and experimentation
* Evaluation on a genuinely held-out test set, with precision/recall/confusion matrix reporting

## Dataset
The dataset consists of road images labeled as:
* Potholes
* Plain roads

Images are resized to 100×100 and loaded in RGB. The dataset currently contains **724 training images** (357 pothole, 367 plain) and **16 held-out test images** (8 per class) that are never used during training.

## Technologies Used
* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* scikit-learn (for evaluation metrics)

## Project Structure
model/ # Saved trained model (.h5)
src/ # Training and prediction scripts
dataset/ # Train and test images, organized by class


## How to Run
1. Install dependencies:

pip install numpy tensorflow keras opencv-python scikit-learn

2. Train the model:

python3 src/main.py

3. Run a prediction on a single image:

python3 src/predictor.py


## Results
The current model achieves **87.5% accuracy** on a held-out test set of 16 images never seen during training. The architecture uses 3 convolutional layers with max pooling, trained for up to 50 epochs with early stopping based on validation loss.

> **Note on evaluation methodology:** an earlier version of this project reported ~60% accuracy, but that figure was computed on a validation split of the *training* data rather than a genuinely separate test set — it did not reflect real generalization performance. This version fixes that by evaluating exclusively on images in `dataset/test/`, which the model never sees during training.

Given the current test set is small (16 images), this accuracy figure should be treated as an early estimate rather than a statistically robust benchmark — see Future Work below.

## Future Work
* Grow the test set well beyond 16 images for a more statistically reliable accuracy measurement
* Add data augmentation (rotation, brightness, flips) to improve robustness given the current dataset size
* Move from binary classification to object detection (e.g., YOLOv8) to localize potholes with bounding boxes rather than classifying whole images
* Build a simple web demo (Flask or Streamlit) for interactive predictions

## Author
Nandika Garg
Reg No: 2427030369
