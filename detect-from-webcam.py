import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for i, (x, y, w, h) in enumerate(faces):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, f'Face #{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.putText(img, f'Total Faces: {len(faces)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Detection', img)
    if cv2.waitKey(30) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
