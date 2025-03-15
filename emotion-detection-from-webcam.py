import cv2
from deepface import DeepFace
import time
from collections import defaultdict
import os
import random
from pygame import mixer

interval = 10

def detect_emotion(image):
    try:
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0].get('dominant_emotion', 'Unknown')
        emotion_scores = analysis[0].get('emotion', {})
        return dominant_emotion, emotion_scores
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "Unknown", {}

def play_music(emotion, music_folder="music_db"):
    emotion_folder = os.path.join(music_folder, emotion.lower())
    print(emotion_folder)
    if os.path.exists(emotion_folder) and os.path.isdir(emotion_folder):
        music_files = [f for f in os.listdir(emotion_folder) if f.endswith('.mp3')]
        if music_files:
            selected_music = random.choice(music_files)
            music_path = os.path.join(emotion_folder, selected_music)
            print(f"Playing: {selected_music} for emotion: {emotion.upper()}")
            mixer.init()
            mixer.music.load(music_path)
            mixer.music.play()
        else:
            print(f"No music files found for emotion: {emotion.upper()}")
    else:
        print(f"No folder found for emotion: {emotion.upper()}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_accumulator = defaultdict(float)
frame_count = 0
start_time = time.time()
current_emotion = None

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

        print("\nScorezzzzzzz:")
        for emotion, avg_score in avg_emotions.items():
            print(f"{emotion.capitalize()}: {avg_score:.2f}")

        print(f"\nAverage emotionnnnn: {dominant_avg_emotion.upper()}")

        if dominant_avg_emotion != current_emotion:
            current_emotion = dominant_avg_emotion
            play_music(current_emotion)

        emotion_accumulator.clear()
        frame_count = 0
        start_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
