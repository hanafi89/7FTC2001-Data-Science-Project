import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 🔹 Load trained model
model = load_model('emotion_model.keras')

# 🔹 MUST match Colab label order
labels = ['angry','happy','neutral','sad','surprise']

# 🔹 Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 🔹 Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔹 Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # 🔥 IMPORTANT — match training preprocessing
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)   # grayscale
        face = cv2.resize(face, (64,64))                # correct size
        face = face / 255.0                             # normalize

        # reshape to match model input
        face = np.reshape(face, (1,64,64,1))

        # 🔹 Predict
        pred = model.predict(face, verbose=0)

        emotion = labels[np.argmax(pred)]
        confidence = np.max(pred)

        text = f"{emotion} ({confidence:.2f})"

        # 🔹 Draw box + text
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()