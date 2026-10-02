import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        # Face embedding
        embedding = face.embedding

        print("Embedding shape:", embedding.shape)
        print("First 10 values:", embedding[:10])

        # Face bounding box
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Embedding", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()