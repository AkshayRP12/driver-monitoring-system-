import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load models
yolo_model = YOLO("models/yolov8n.pt")
drowsy_model = load_model("models/drowsiness_model.h5")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- YOLO (Phone Detection) --------
    results = yolo_model(frame)
    phone_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if label == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Phone", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # -------- Drowsiness Detection --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_closed = False

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (64, 64)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        prediction = drowsy_model.predict(face_resized, verbose=0)

        if prediction[0][0] < 0.5:
            eyes_closed = True
            label = "Drowsy"
            color = (0, 0, 255)
        else:
            label = "Awake"
            color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # -------- Decision Logic --------
    if eyes_closed and phone_detected:
        cv2.putText(frame, "HIGH RISK!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    elif eyes_closed:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    elif phone_detected:
        cv2.putText(frame, "DISTRACTION ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show output
    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()