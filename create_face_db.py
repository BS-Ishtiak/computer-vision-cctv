import os
import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis

FACES_DIR = "faces"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# 1. Load InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. People = one subfolder per person inside faces/ (e.g. ishtiak, ronaldo-cr7)
people = sorted(
    d for d in os.listdir(FACES_DIR)
    if os.path.isdir(os.path.join(FACES_DIR, d))
)
print("People found:", people)

# 3. Extract embeddings from every image of every person
embeddings = []
labels = []

for person_id, person_name in enumerate(people):
    person_folder = os.path.join(FACES_DIR, person_name)
    print(f"\nProcessing {person_name}...")

    for image_file in sorted(os.listdir(person_folder)):
        if not image_file.lower().endswith(IMAGE_EXTS):
            continue

        image_path = os.path.join(person_folder, image_file)

        # InsightFace needs a BGR image array, not a file path
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Could not read: {image_file}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"  No face found in: {image_file}")
            continue

        # If several faces are in the photo, keep the largest one
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Normalized embedding -> inner product equals cosine similarity
        embeddings.append(face.normed_embedding)
        labels.append(person_id)
        print(f"  OK {image_file} - embedding shape: {face.normed_embedding.shape}")

if len(embeddings) == 0:
    print("\nNo embeddings extracted! Check your images.")
    exit()

# 4. Convert to NumPy matrix
embeddings = np.array(embeddings).astype("float32")
labels = np.array(labels).astype("int32")

print("\nDatabase shape:", embeddings.shape)

# 5. Create FAISS index (inner product on normalized vectors = cosine similarity)
index = faiss.IndexFlatIP(512)

# 6. Add embeddings to FAISS
index.add(embeddings)
print("Number of faces in FAISS:", index.ntotal)

# 7. Save index, labels, and names
faiss.write_index(index, "face_database.index")
np.save("face_labels.npy", labels)
np.save("face_names.npy", np.array(people))

print("FAISS database saved!")
