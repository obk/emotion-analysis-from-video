import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd

# Constants for optimization
FRAME_SKIP = 5  # Skip frames to reduce noise
EMOTION_CONFIDENCE_THRESHOLD = 0.8  # Ignore low-confidence predictions
TEMPORAL_WINDOW_SIZE = 5  # Window size for temporal smoothing

def preprocess_face(face_image):
    """Normalize brightness/contrast and resize face."""
    # Normalize brightness/contrast (optional)
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 30   # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(face_image, alpha=alpha, beta=beta)

    # Resize face to a fixed size
    target_size = (224, 224)  # Size expected by DeepFace
    resized_face = cv2.resize(adjusted, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_face

def hybrid_face_detection(frame):
    """Hybrid approach with OpenCV Haar Cascades + DeepFace."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Use Haar Cascades for initial detection
    faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Refine detections with DeepFace
    refined_faces = []
    for (x, y, w, h) in faces_haar:
        roi = frame[y:y+h, x:x+w]
        result = DeepFace.extract_faces(roi, enforce_detection=False)
        if len(result) > 0:
            refined_faces.append((x, y, w, h))
    
    return refined_faces

def run_analysis(video_path):
    try:
        print(f'Running analysis on {video_path}...')
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        emotions = {}
        emotions_per_second = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce noise and improve speed
            if frame_count % FRAME_SKIP != 0:
                frame_count += 1
                continue
            
            roi = frame[:frame.shape[0]//2, :]
            refined_faces = hybrid_face_detection(roi)
            
            for (x, y, w, h) in refined_faces:
                face_image = roi[y:y+h, x:x+w]
                
                # Preprocess the face
                processed_face = preprocess_face(face_image)
                
                try:
                    result_emotion = DeepFace.analyze(processed_face, actions=['emotion'], enforce_detection=False)[0]
                    
                    dominant_emotion = result_emotion['dominant_emotion']
                    confidence = result_emotion['emotion'][dominant_emotion]
                    
                    # Ignore low-confidence predictions
                    if confidence < EMOTION_CONFIDENCE_THRESHOLD:
                        continue
                    
                    print(f"Frame: {frame_count} Emotion: {dominant_emotion} (Confidence: {confidence})")
                    emotions[frame_count] = dominant_emotion
                    current_second = int(frame_count / fps)
                    
                    if current_second not in emotions_per_second:
                        emotions_per_second[current_second] = {}
                    if dominant_emotion not in emotions_per_second[current_second]:
                        emotions_per_second[current_second][dominant_emotion] = []
                    
                    emotions_per_second[current_second][dominant_emotion].append(confidence)
                
                except Exception as e:
                    print(f"Error analyzing frame {frame_count}: {e}")

            frame_count += 1

        cap.release()

        # Temporal smoothing
        smoothed_emotions = {}
        for sec, emotion_data in emotions_per_second.items():
            smoothed_emotion = max(emotion_data.keys(), key=lambda k: sum(emotion_data[k]) / len(emotion_data[k]))
            smoothed_emotions[sec] = smoothed_emotion

        return smoothed_emotions

    except Exception as e:
        print(f"Error running analysis: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python app.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = run_analysis(video_path)
    
    if result is not None:
        for sec, emotion in result.items():
            print(f"Second {sec}: {emotion}")
