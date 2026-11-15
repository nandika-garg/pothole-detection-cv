# evaluate.py
# Loads the SAVED model and evaluates it on the test folders. No training,
# so the result is the same every time you run it.
# Run from the project root (inside the venv):
#   python3 src/evaluate.py

import glob
import math
import os

import numpy as np
import cv2
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 100
CLASS_NAMES = ["plain", "pothole"]  # 0 = plain, 1 = pothole (same as training)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pothole_model.h5")
TEST_SETS = {
    "Main test set (dataset/test)": os.path.join(BASE_DIR, "dataset", "test"),
    "Original 16 images (dataset/test_old)": os.path.join(BASE_DIR, "dataset", "test_old"),
}


def load_folder(folder):
    """Return (images, labels, paths) using the SAME preprocessing as training."""
    images, labels, paths = [], [], []
    unreadable = 0
    for label, name in enumerate(CLASS_NAMES):
        for path in sorted(glob.glob(os.path.join(folder, name, "*"))):
            if path.endswith(".DS_Store"):
                continue
            img = cv2.imread(path)
            if img is None:
                unreadable += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
            paths.append(path)
    if unreadable:
        print(f"  (skipped {unreadable} unreadable files)")
    if not images:
        return None, None, None
    x = np.array(images).reshape(len(images), IMG_SIZE, IMG_SIZE, 3) / 255.0
    return x, np.array(labels), paths


def evaluate(title, folder, model):
    print("=" * 60)
    print(title)
    print("=" * 60)
    if not os.path.isdir(folder):
        print("  Folder not found, skipping.\n")
        return
    x, y, paths = load_folder(folder)
    if x is None:
        print("  No images found, skipping.\n")
        return

    counts = {name: int(np.sum(y == i)) for i, name in enumerate(CLASS_NAMES)}
    print(f"Images: {len(y)}  ({counts['pothole']} pothole, {counts['plain']} plain)")

    pred = np.argmax(model.predict(x, verbose=0), axis=1)
    acc = float(np.mean(pred == y))
    margin = 1.96 * math.sqrt(acc * (1 - acc) / len(y))
    print(f"\nAccuracy: {acc:.1%}  (95% margin of error: +/- {margin:.1%})")

    print("\nClassification report:")
    print(classification_report(y, pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted; order: plain, pothole):")
    print(confusion_matrix(y, pred, labels=[0, 1]))

    wrong = [
        (os.path.relpath(p, BASE_DIR), CLASS_NAMES[t], CLASS_NAMES[q])
        for p, t, q in zip(paths, y, pred)
        if t != q
    ]
    print(f"\nMisclassified: {len(wrong)}")
    for p, t, q in wrong[:15]:
        print(f"  {p}: true={t}, predicted={q}")
    if len(wrong) > 15:
        print(f"  ... and {len(wrong) - 15} more (full list saved to misclassified.txt)")
    out = os.path.join(BASE_DIR, "misclassified.txt")
    with open(out, "a") as f:
        f.write(f"# {title}\n")
        for p, t, q in wrong:
            f.write(f"{p}\ttrue={t}\tpredicted={q}\n")
    print()


if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model not found at {MODEL_PATH}. Run src/main.py first.")

# Start a fresh misclassified.txt for this run
open(os.path.join(BASE_DIR, "misclassified.txt"), "w").close()

model = load_model(MODEL_PATH, compile=False)
for title, folder in TEST_SETS.items():
    evaluate(title, folder, model)