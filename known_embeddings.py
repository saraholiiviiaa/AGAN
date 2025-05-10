import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

from layers.ghost_module import GhostModule
from layers.aca import AdaptiveContextualAttention as ACA
from layers.iaca import InputAwareComplexityAdjustment as IACA
from layers.hal import HybridActivationLayer as HAL
from arcface import ArcMarginProduct

IMG_SIZE = 112
MODEL_PATH = "agan_small_arcface.h5"
DATASET_PATH = "/Users/saraholivia/Desktop/university/big_data_final/GhostFaceNets-main/faces_dataset"
EMBEDDINGS_PATH = "/Users/saraholivia/Desktop/university/big_data_final/GhostFaceNets-main/known_embeddings.pkl"

# === LOAD TRAINED MODEL WITH CUSTOM OBJECTS ===
custom_objects = {
    'GhostModule': GhostModule,
    'AdaptiveContextualAttention': ACA,
    'InputAwareComplexityAdjustment': IACA,
    'HybridActivationLayer': HAL,
    'ArcMarginProduct': ArcMarginProduct
}

model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
embedding_model = tf.keras.Model(inputs=model.input[0], outputs=model.get_layer("dense").output)
embedding_model = tf.keras.Sequential([embedding_model, tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))])


# === PROCESS FOLDER ===
embeddings = []
labels = []
label_map = {}
label_counter = 0

for person_name in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_path):
        continue

    if person_name not in label_map:
        label_map[person_name] = label_counter
        label_counter += 1

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            emb = embedding_model.predict(img)[0]
            embeddings.append(emb)
            labels.append(label_map[person_name])
        except:
            print(f"Failed to process {img_path}")

# === SAVE ===
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump({"embeddings": embeddings, "labels": labels, "label_map": label_map}, f)

print(f"✅ Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH}")
