import cv2
from deepface import DeepFace

cap = cv2.VideoCapture("video.mp4")

# Set the frame rate to 10 frames per second
cap.set(cv2.CAP_PROP_FPS, 10)

prev_emotion = None
prev_time = 0
emotion_duration_threshold = 0.5
current_emotion_duration = 0
emotion_changes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotions in the frame
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Get the dominant emotion
    dominant_emotion = result[0]['dominant_emotion']

    # Get the current time
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # Check if the emotion has changed
    if dominant_emotion != prev_emotion:
        # Check if the previous emotion lasted for at least the threshold duration
        if current_emotion_duration >= emotion_duration_threshold:
            emotion_changes.append({'start_time': prev_time, 'end_time': current_time, 'emotion': prev_emotion})
        # Reset the current emotion duration
        current_emotion_duration = 0
        # Update the previous emotion and time
        prev_emotion = dominant_emotion
        prev_time = current_time
    else:
        # Increment the current emotion duration
        current_emotion_duration += 1 / cap.get(cv2.CAP_PROP_FPS)

cap.release()

# Print out the emotion changes
for change in emotion_changes:
    print(f"{change['start_time']:.2f} to {change['end_time']:.2f}: {change['emotion']}")
