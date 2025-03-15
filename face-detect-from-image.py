import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('test.jpg')
img = cv2.resize(img, (800, 600))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for i, (x, y, w, h) in enumerate(faces):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    cv2.putText(img, f'Face #{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

overlay = img.copy()
cv2.rectangle(overlay, (10, 10), (200, 50), (0, 0, 0), -1)
img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
cv2.putText(img, f'Total Faces: {len(faces)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.imshow('Enhanced Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
