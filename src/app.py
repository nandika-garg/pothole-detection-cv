# app.py
# Streamlit interface for the pothole detection CNN.
# Run from the project root with:  streamlit run src/app.py

import os

import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

IMG_SIZE = 100
CLASS_NAMES = ["Plain Road", "Pothole"]  # index 0 = plain, 1 = pothole (same as training)

# Build the model path from this file's location so it works from any folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pothole_model.h5")


@st.cache_resource
def get_model():
    # compile=False: we only need predictions, not training settings
    return load_model(MODEL_PATH, compile=False)


def preprocess(img_rgb):
    # Must match training exactly: resize -> scale to 0-1 -> shape (1, 100, 100, 3)
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 3)


st.set_page_config(page_title="Pothole Detection", page_icon="🛣️")
st.title("Pothole Detection")
st.write(
    "Upload a road image and the CNN will classify it as a **pothole** "
    "or a **plain road**."
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Run src/main.py to train and save it first.")
    st.stop()

model = get_model()

uploaded = st.file_uploader("Choose a road image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read that image. Try a different file.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded image")

        probs = model.predict(preprocess(img_rgb), verbose=0)[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])

        if pred == 1:
            st.error(f"Prediction: Pothole ({confidence:.1%} confidence)")
        else:
            st.success(f"Prediction: Plain Road ({confidence:.1%} confidence)")

        st.subheader("Class probabilities")
        for name, p in zip(CLASS_NAMES, probs):
            st.write(f"{name}: {float(p):.1%}")
            st.progress(float(p))

        st.caption(
            "Prototype model trained on a small dataset. It tends to flag plain "
            "roads as potholes rather than miss real ones."
        )