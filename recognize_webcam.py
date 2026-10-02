import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis

# Cosine similarity at or above this = same person
THRESHOLD = 0.4

# 1. Load InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load the database built by create_face_db.py
index = faiss.read_index("face_database.index")
labels = np.load("face_labels.npy")
names = np.load("face_names.npy")
print(f"Loaded {index.ntotal} faces for: {list(names)}")

# 3. Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    faces = app.get(frame)

    for face in faces:
        # 4. Search for the closest face in the database
        query = face.normed_embedding.reshape(1, -1).astype("float32")
        scores, ids = index.search(query, 1)
        score = float(scores[0][0])

        if score >= THRESHOLD:
            name = str(names[labels[ids[0][0]]])
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # 5. Draw box and label
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
