# check_duplicates.py
# Finds test images that are byte-for-byte copies of training images
# (or of each other). Run from the project root:
#   python3 src/check_duplicates.py
#
# This only READS files. It never deletes anything.

import hashlib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(folder):
    """Return {hash: [paths]} for every image under folder."""
    found = {}
    for root, _, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            path = os.path.join(root, name)
            found.setdefault(file_hash(path), []).append(path)
    return found


def rel(path):
    return os.path.relpath(path, BASE_DIR)


train = collect(TRAIN_DIR)
test = collect(TEST_DIR)

train_count = sum(len(v) for v in train.values())
test_count = sum(len(v) for v in test.values())
print(f"Train images scanned: {train_count}")
print(f"Test images scanned:  {test_count}")

# 1. Test images that also exist in train (this is the serious one)
leaks = [(h, paths) for h, paths in test.items() if h in train]
print(f"\nTest images that are also in train: {sum(len(p) for _, p in leaks)}")
for h, paths in leaks:
    for p in paths:
        print(f"  TEST {rel(p)}")
    for p in train[h]:
        print(f"     = TRAIN {rel(p)}")

# 2. Duplicates inside the test set itself (inflates the count)
inner = [paths for paths in test.values() if len(paths) > 1]
print(f"\nDuplicate groups inside the test set: {len(inner)}")
for paths in inner:
    print("  " + "  ==  ".join(rel(p) for p in paths))

print(
    "\nNote: this only catches identical files. A resized or re-saved copy "
    "of the same photo will not be detected."
)