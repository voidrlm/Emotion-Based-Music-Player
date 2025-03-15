import cv2
from deepface import DeepFace
import time
from collections import defaultdict

interval = 5

def detect_emotion(image):
    try:
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0].get('dominant_emotion', 'Unknown')
        emotion_scores = analysis[0].get('emotion', {})
        return dominant_emotion, emotion_scores
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "Unknown", {}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_accumulator = defaultdict(float)
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    dominant_emotion, emotion_scores = detect_emotion(rgb_frame)

    for emotion, score in emotion_scores.items():
        emotion_accumulator[emotion] += score
    frame_count += 1

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    current_time = time.time()
    if current_time - start_time >= interval:
        avg_emotions = {emotion: score/frame_count for emotion,
                        score in emotion_accumulator.items()}

        dominant_avg_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]

        print("\nAverage emotions over last 5 seconds:")
        for emotion, avg_score in avg_emotions.items():
            print(f"{emotion.capitalize()}: {avg_score:.2f}")

        print(f"\nDominant Average Emotion: {dominant_avg_emotion.upper()}")

        emotion_accumulator.clear()
        frame_count = 0
        start_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
