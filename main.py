from flask import Flask, jsonify, request
import os
import cv2
import shutil
import pandas as pd
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'emocall-edd87.appspot.com'
})

@app.route('/')
def home():
    return 'Server is running.'

@app.route('/start', methods=['GET','POST'])
def start_analysis():
    result = run_analysis()
    return jsonify(result)

def run_analysis():
    try:
        print('Running analysis...')

        bucket = storage.bucket()
        blobs = bucket.list_blobs()
        files = {}

        for blob in blobs:
            files[blob.name] = blob.updated

        latest_file = max(files, key=files.get)
        blob = bucket.blob(latest_file)
        blob.download_to_filename(f'./{latest_file}')

        print(f'The latest uploaded file is: {latest_file} and it has been downloaded.')

        cap = cv2.VideoCapture(f'./{latest_file}')
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        emotions = {}
        emotions_per_second = {}

        while True:
            ret, frame = cap.read()
            if ret:
                roi = frame[:frame.shape[0]//2, :]
                result = DeepFace.extract_faces(roi, enforce_detection=False)
                if len(result) > 0:
                    face_coords = result[0]['facial_area']
                    x, y, w, h = face_coords['x'], face_coords['y'], face_coords['w'], face_coords['h']
                    face_image = roi[y:y+h, x:x+w]
                    cv2.imshow("Frame", face_image)
                    result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent = True)
                    print("Frame: {}".format(frame_count) + " Emotion: {}".format(result[0]['dominant_emotion']))
                    emotions[frame_count] = result[0]['dominant_emotion']
                    current_second = int(frame_count / fps)
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

        result = {}
        prev_emotion = None
        emotion_start = 0

        for second in emotions_per_second:
            dominant_emotion = max(emotions_per_second[second], key=emotions_per_second[second].get)
            result[str(second)] = {
                "dominant_emotion": dominant_emotion,
                "previous_emotion": prev_emotion
            }
            if prev_emotion is not None and dominant_emotion != prev_emotion:
                result[str(second)]["emotion_change_detected"] = True
                if (second - emotion_start) > 1:
                    result[str(second)]["emotion_duration"] = second - emotion_start
                emotion_start = second
            prev_emotion = dominant_emotion

        cap.release()
        cv2.destroyAllWindows()
        return result

    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(port=5000)
