import os
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from agan_model import build_agan_model
from arcface import ArcMarginProduct


# === CONFIG ===
IMAGE_SIZE = (112, 112)
EMBEDDING_DIM = 512
MODEL_WEIGHTS_PATH = "checkpoints/agan_trained_arcface/variables/variables"  # TF SavedModel
KNOWN_DIR = "test_faces/known"
UNKNOWN_DIR = "test_faces/unknown"
THRESHOLD = 0.5

# === Load architecture and weights ===
print("✅ Loading embedding model...")
base_model = build_agan_model(input_shape=IMAGE_SIZE + (3,), embedding_dim=EMBEDDING_DIM)
input_image = tf.keras.Input(shape=IMAGE_SIZE + (3,), name="input_image")
embedding_output = base_model(input_image)
embedding_model = tf.keras.Model(inputs=input_image, outputs=embedding_output)
embedding_model.load_weights(MODEL_WEIGHTS_PATH).expect_partial()

# === Helper: Load and preprocess image ===
def preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        img = np.array(img) / 255.0
        return img
    except Exception as e:
        print(f"⚠️ Could not process image: {img_path} ({e})")
        return None

# === Build known face embeddings ===
print("🔎 Building known face embeddings...")
known_embeddings = []
known_labels = []

for img_file in sorted(Path(KNOWN_DIR).glob("*")):
    if not img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        continue
    img = preprocess_image(str(img_file))
    if img is None:
        continue
    embedding = embedding_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    known_embeddings.append(embedding)
    known_labels.append(img_file.stem)

# === Evaluate unknown faces ===
print("🧪 Evaluating unknown faces...")
correct = 0
total = 0

for img_file in sorted(Path(UNKNOWN_DIR).glob("*")):
    if not img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        continue
    img = preprocess_image(str(img_file))
    if img is None:
        continue
    unknown_embedding = embedding_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

    sims = cosine_similarity([unknown_embedding], known_embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_label = known_labels[best_idx]
    similarity = sims[best_idx]

    actual_label = img_file.stem
    is_correct = actual_label == best_label
    print(f"🧠 {img_file.name} → Predicted: {best_label} | Actual: {actual_label} | {'✅ True' if is_correct else '❌ False'} (Similarity: {similarity:.2f})")

    total += 1
    if is_correct:
        correct += 1

# === Final Accuracy ===
accuracy = correct / total if total > 0 else 0.0
print(f"✅ Evaluation complete: {total} tested | ✅ Correct: {correct} | 🎯 Accuracy: {accuracy:.2%}")
