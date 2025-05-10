from deepface import DeepFace
import os

# === CONFIG ===
TEST_IMAGES = [
    ("test_faces/known/Angelina Jolie/01.jpg", "test_faces/known/Angelina Jolie/02.jpg"),  # same
    ("test_faces/known/Hugh Jackman/01.jpg", "test_faces/unknown/random1.jpeg"),              # different
    ("test_faces/known/Megan Fox/01.jpg", "test_faces/known/Megan Fox/01.jpg")               # same
]

# === RUN COMPARISONS ===
for img1_path, img2_path in TEST_IMAGES:
    result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name="ArcFace")
    is_same = result['verified']
    score = result['distance']
    print(f"🔍 {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} → Match: {is_same}, Distance: {score:.4f}")
