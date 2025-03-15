import cv2
from deepface import DeepFace
import time
from collections import defaultdict
import os
import random
from pygame import mixer

interval = 10  
crossfade_duration = 5  

def detect_attributes(image):
    try:
        analysis = DeepFace.analyze(image, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
        
        dominant_emotion = analysis[0].get('dominant_emotion', 'Unknown')
        emotion_scores = analysis[0].get('emotion', {})

        age = analysis[0].get('age', 'Unknown')
        gender = analysis[0].get('dominant_gender', 'Unknown')
        race = analysis[0].get('dominant_race', 'Unknown')

        return dominant_emotion, emotion_scores, age, gender, race
    except Exception as e:
        print(f"Error analyzing attributes: {e}")
        return "Unknown", {}, "Unknown", "Unknown", "Unknown"

def play_music_with_crossfade(new_emotion, current_channel, music_folder="music_db"):
    emotion_folder = os.path.join(music_folder, new_emotion.lower())
    
    if os.path.exists(emotion_folder) and os.path.isdir(emotion_folder):
        music_files = [f for f in os.listdir(emotion_folder) if f.endswith('.mp3')]
        if music_files:
            selected_music = random.choice(music_files)
            music_path = os.path.join(emotion_folder, selected_music)
            
            mixer.init()
            new_channel = mixer.Channel(1)
            new_channel.set_volume(0)
            new_sound = mixer.Sound(music_path)
            new_channel.play(new_sound)
            steps = 100  
            delay_per_step = crossfade_duration / steps  

            for i in range(steps):
                current_channel.set_volume(1 - i / steps) 
                new_channel.set_volume(i / steps)        
                time.sleep(delay_per_step)

            return new_channel 
        else:
            print(f"No music files found for emotion: {new_emotion.upper()}")
    else:
        print(f"No folder found for emotion: {new_emotion.upper()}")
    
    return current_channel


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emotion_accumulator = defaultdict(float)
frame_count = 0
start_time = time.time()
current_emotion = None


mixer.init()
current_channel = mixer.Channel(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    dominant_emotion, emotion_scores, age, gender, race = detect_attributes(rgb_frame)

    for emotion, score in emotion_scores.items():
        emotion_accumulator[emotion] += score
    frame_count += 1

 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {age}', (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Gender: {gender}', (x, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'Race: {race}', (x, y - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow('Emotion Detection', frame)

    current_time = time.time()
    if current_time - start_time >= interval:
        avg_emotions = {emotion: score/frame_count for emotion, score in emotion_accumulator.items()}
        
        if avg_emotions:
            dominant_avg_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
        else:
            dominant_avg_emotion = "neutral"

        print("\nEmotion Scores:")
        for emotion, avg_score in avg_emotions.items():
            print(f"{emotion.capitalize()}: {avg_score:.2f}")

        print(f"\nCurrent Vibe: {dominant_avg_emotion.upper()}")
        print(f"Age: {age}, Gender: {gender}, Race: {race}")

 
        if dominant_avg_emotion != current_emotion:
            current_emotion = dominant_avg_emotion
            current_channel = play_music_with_crossfade(current_emotion, current_channel)


        emotion_accumulator.clear()
        frame_count = 0
        start_time = current_time


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
