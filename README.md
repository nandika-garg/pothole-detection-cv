# Pothole Detection using Computer Vision

## Project Description
This project detects potholes on road images using a Convolutional Neural Network (CNN). 
It classifies images into pothole and plain road.

## Features
- Image classification using CNN
- Detects potholes from input images
- Simple and lightweight model

## Dataset
The dataset consists of images of:
- Potholes
- Plain roads

Images are resized and converted to grayscale.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## Project Structure
`model/` `src/` `dataset/` `index.html`

## How to Run

1. Install dependencies:
pip install numpy tensorflow keras opencv-python

2. Run training:
python3 src/main.py

3. Run prediction:
python3 src/predictor.py

## Results
Model achieves around 60% accuracy on test data.

## Future Work
- Improve dataset
- Use YOLO for real-time detection

## Author
Nandika Garg  
Reg No: 2427030369



