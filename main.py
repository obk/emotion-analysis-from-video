import os
import cv2
import shutil
import pandas as pd
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
  'storageBucket': 'emocall-a5798.appspot.com'
})

bucket = storage.bucket()

# List all files in the bucket
blobs = bucket.list_blobs()

# Initialize an empty dictionary to store file names and their timestamps
files = {}

for blob in blobs:
    # Add blob name and timestamp to dictionary
    files[blob.name] = blob.updated

# Find the name of the file that has the latest timestamp
latest_file = max(files, key=files.get)

# Download the latest file
blob = bucket.blob(latest_file)
blob.download_to_filename(f'./{latest_file}')

print(f'The latest uploaded file is: {latest_file} and it has been downloaded.')

print(latest_file)

# Load the video
cap = cv2.VideoCapture(f'./{latest_file}')

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0
emotions = {}  # dictionary to store dominant emotions for each frame
emotions_per_second = {}  # dictionary to store dominant emotions for each second 

while True:
    ret, frame = cap.read()
    if ret:
        # Define the region of interest for cropping - take only top half of the video
        roi = frame[:frame.shape[0]//2, :]

        # Use deepface to detect faces in the video for each frame
        result = DeepFace.extract_faces(roi, enforce_detection=False)
        # Check if a face was detected in the frame
        if len(result) > 0:
            # Get the coordinates of the detected face
            face_coords = result[0]['facial_area']
            x, y, w, h = face_coords['x'], face_coords['y'], face_coords['w'], face_coords['h']
            # Crop the face from the original frame
            face_image = roi[y:y+h, x:x+w]
            cv2.imshow("Frame", face_image)
            # Detect face from croped image
            result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent = True)
            print("Frame: {}".format(frame_count) + " Emotion: {}".format(result[0]['dominant_emotion']))
            # Store the dominant emotion for each frame
            emotions[frame_count] = result[0]['dominant_emotion']
            # Calculate the current second based on the frame count and FPS
            current_second = int(frame_count / fps)
            # Add the dominant emotion to the dictionary for the current second
            if current_second not in emotions_per_second:
                emotions_per_second[current_second] = {}
            if emotions[frame_count] not in emotions_per_second[current_second]:
                emotions_per_second[current_second][emotions[frame_count]] = 0
            emotions_per_second[current_second][emotions[frame_count]] += 1
            frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

prev_emotion = None  # initialize previous emotion as None
for second in emotions_per_second:
    dominant_emotion = max(emotions_per_second[second], key=emotions_per_second[second].get)
    if prev_emotion is not None and dominant_emotion != prev_emotion:
        print("Emotion changed from {} to {} at second {}".format(prev_emotion, dominant_emotion, second))
    prev_emotion = dominant_emotion  # update previous emotion for next iteration

# Release the VideoCapture object and close any windows
cap.release()
cv2.destroyAllWindows()