import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/drowsiness_model.h5")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Resize for model
        face_resized = cv2.resize(face, (64, 64))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        prediction = model.predict(face_resized)

        label = "Open Eyes" if prediction[0][0] > 0.5 else "Closed Eyes"

        color = (0, 255, 0) if label == "Open Eyes" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()