import cv2
import os

# 🔹 CHANGE THESE BEFORE RUNNING
emotion = "surprised"        # change each time
session = "session5"     # change session
save_path = f"emotion_dataset/{session}/{emotion}"

# create folder if not exist
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
max_images = 200

print("Press SPACE to capture, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"{emotion} - {count}/{max_images}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # SPACE to capture
        filename = f"{save_path}/{emotion}_{count}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Saved: {filename}")

    elif key == ord('q'):
        break

    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()