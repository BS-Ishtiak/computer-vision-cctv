import cv2
from insightface.app import FaceAnalysis

# Create InsightFace application
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

# Prepare the model
app.prepare(ctx_id=0, det_size=(640, 640))

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read from webcam")
        break

    # Detect faces
    faces = app.get(frame)

    # Draw information for each face
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("InsightFace - Face Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()