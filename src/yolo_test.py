from ultralytics import YOLO
import cv2

# Load YOLO model (downloads automatically first time)
model = YOLO("models/yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    phone_detected = False

    # Loop through results
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Check if detected object is a phone
            if label == "cell phone":
                phone_detected = True

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Put label
                cv2.putText(frame, "Phone Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display alert on screen
    if phone_detected:
        cv2.putText(frame, "DISTRACTION ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show output
    cv2.imshow("YOLO Phone Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()