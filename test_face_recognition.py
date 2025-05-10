import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
from numpy.linalg import norm

# 🔧 Fix deserialization error by registering the custom activation
def hard_sigmoid_torch(x):
    return keras.backend.clip((x + 3) / 6, 0.0, 1.0)

keras.utils.get_custom_objects()["hard_sigmoid_torch"] = hard_sigmoid_torch
keras.utils.get_custom_objects()["function"] = hard_sigmoid_torch  # for legacy

# === Load model ===
print("✅ Loading model...")
model = load_model("checkpoints/ghostnet_130_basic_model_latest.keras", compile=False)
print("✅ Model loaded!")


# === Preprocessing helper ===
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Could not read image: {path}")
        return None
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# === Encode image using the model ===
def get_embedding(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return None
    embedding = model.predict(img)[0]
    return embedding / norm(embedding)

# === Load known faces ===
print("🔍 Loading known faces...")
known_embeddings = {}
known_dir = "/Users/saraholivia/Desktop/university/big_data_final/GhostFaceNets-main/test_faces/known"
for fname in os.listdir(known_dir):
    path = os.path.join(known_dir, fname)
    embedding = get_embedding(path)
    if embedding is not None:
        known_embeddings[fname] = embedding
print(f"✅ Loaded {len(known_embeddings)} known faces.")

# === Load and compare unknown faces ===
unknown_dir = "/Users/saraholivia/Desktop/university/big_data_final/GhostFaceNets-main/test_faces/unknown"
threshold = 0.5  # You can tune this based on performance
print("\n🔎 Matching unknown faces:\n")

for fname in os.listdir(unknown_dir):
    path = os.path.join(unknown_dir, fname)
    unknown_embedding = get_embedding(path)
    if unknown_embedding is None:
        continue

    best_match = None
    best_score = -1

    for known_name, known_embedding in known_embeddings.items():
        sim = np.dot(unknown_embedding, known_embedding)
        if sim > best_score:
            best_score = sim
            best_match = known_name

    if best_score > threshold:
        print(f"🟢 {fname} matches with {best_match} (Similarity: {best_score:.2f})")
    else:
        print(f"🔴 {fname} is unknown (Highest Similarity: {best_score:.2f})")
