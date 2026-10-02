import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# YOLO PERSON DETECTION - Beginner Guide
# ============================================================================
# This script uses YOLOv8 to detect people in a webcam stream
# YOLOv8 is a fast, accurate object detection model
# ============================================================================

# Step 1: Load the YOLO model
# YOLOv8n is the "nano" version - smallest and fastest
# Other sizes: 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
# Larger = more accurate but slower
print("Loading YOLO model... (this may take a moment)")
model = YOLO("yolov8n.pt")  # 'pt' = PyTorch format
print("✓ Model loaded successfully!")

# Step 2: Open the webcam
# cv2.VideoCapture(0) means use the default webcam (camera #0)
# If you have multiple cameras, try 1, 2, 3, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("✓ Webcam opened successfully!")
print("Press 'q' to quit\n")

# Step 3: Main loop - process each frame from the webcam
frame_count = 0
detected_count = 0

while True:
    # Read one frame from the webcam
    ok, frame = cap.read()

    if not ok:
        print("❌ Error reading frame")
        break

    frame_count += 1

    # ========================================================================
    # DETECTION: Run YOLO on the frame
    # ========================================================================
    # model(frame) = run object detection on the frame
    # The result contains: boxes, confidence scores, class labels
    results = model(frame, verbose=False)  # verbose=False = less console output

    # Extract the boxes from results
    # boxes = [x1, y1, x2, y2, confidence, class_id]
    # x1, y1 = top-left corner of box
    # x2, y2 = bottom-right corner of box
    # confidence = how sure (0-1, higher = more sure)
    # class_id = object type (0 = person, 1 = car, etc.)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Convert to numpy array
    classes = results[0].boxes.cls.cpu().numpy()  # Class IDs
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # ========================================================================
    # FILTER: Keep only "person" detections (class 0 in YOLO)
    # ========================================================================
    # YOLO class 0 = person, so we filter for that
    person_detections = []

    for i, class_id in enumerate(classes):
        if int(class_id) == 0:  # 0 = person class in YOLO
            x1, y1, x2, y2 = boxes[i].astype(int)
            confidence = float(confidences[i])
            person_detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': confidence
            })

    detected_count += len(person_detections)

    # ========================================================================
    # DRAW: Draw boxes and labels on the frame
    # ========================================================================
    for idx, detection in enumerate(person_detections):
        x1, y1, x2, y2 = detection['box']
        confidence = detection['confidence']

        # Draw green rectangle around each detected person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write label with person number and confidence
        label = f"Person #{idx + 1} ({confidence:.2f})"  # e.g., "Person #1 (0.95)"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),  # Text position (slightly above the box)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            0.6,  # Font size
            (0, 255, 0),  # Color (Green)
            2  # Line thickness
        )

    # Add frame info at top-left
    info_text = f"Frame: {frame_count} | People detected: {len(person_detections)}"
    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),  # White
        2
    )

    # ========================================================================
    # DISPLAY: Show the frame in a window
    # ========================================================================
    cv2.imshow("YOLO Person Detection", frame)

    # ========================================================================
    # INPUT: Wait for user input
    # ========================================================================
    # cv2.waitKey(1) = wait 1 millisecond for a key press
    # If user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(f"\n✓ Closed by user")
        print(f"  Total frames: {frame_count}")
        print(f"  Total people detected: {detected_count}")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✓ Webcam closed. Program ended.")
