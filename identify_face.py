import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity

# === Import custom layers ===
from layers.ghost_module import GhostModule
from layers.aca import AdaptiveContextualAttention as ACA
from layers.iaca import InputAwareComplexityAdjustment as IACA
from layers.hal import HybridActivationLayer as HAL
from arcface import ArcMarginProduct

# === Model + Embedding Setup ===
MODEL_PATH = "/Users/saraholivia/Desktop/university/big_data_final/GhostFaceNets-main/agan_small_arcface.h5"

custom_objects = {
    'GhostModule': GhostModule,
    'AdaptiveContextualAttention': ACA,
    'InputAwareComplexityAdjustment': IACA,
    'HybridActivationLayer': HAL,
    'ArcMarginProduct': ArcMarginProduct
}

model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)

# Extract L2-normalized embedding model
embedding_model = tf.keras.Model(inputs=model.input[0], outputs=model.get_layer("dense").output)
embedding_model = tf.keras.Sequential([
    embedding_model,
    tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
])

# === Test Image Comparison Config ===
TEST_IMAGES = [
    ("test_faces/known/Angelina Jolie/01.jpg", "test_faces/known/Angelina Jolie/02.jpg"),   # same person
    ("test_faces/known/Hugh Jackman/01.jpg", "test_faces/unknown/random1.jpeg"),            # different people
    ("test_faces/known/Megan Fox/01.jpg", "test_faces/known/Megan Fox/01.jpg")              # same person
]

def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, (112, 112))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# === Run Comparisons ===
for img1_path, img2_path in TEST_IMAGES:
    img1 = load_and_preprocess(img1_path)
    img2 = load_and_preprocess(img2_path)

    emb1 = embedding_model.predict(img1)[0]
    emb2 = embedding_model.predict(img2)[0]

    sim = cosine_similarity([emb1], [emb2])[0][0]
    print(f"🔍 {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} → Similarity: {sim:.4f}")
