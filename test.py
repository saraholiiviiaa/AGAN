import pickle
import numpy as np

# Try loading the embeddings
with open("known_embeddings.pkl", "rb") as f:
    known = pickle.load(f)

print(f"✅ Loaded {len(known)} identities from known_embeddings.pkl")

# Show the shape of one embedding to confirm structure
for name, emb in known.items():
    print(f"🔎 {name} → shape: {np.array(emb).shape}")
    break
