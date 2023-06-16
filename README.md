Emotion Detection from Video Script
===================================

This Python script is designed to download a video file from a Firebase storage bucket and perform real-time emotion detection on faces present in the video. It uses the DeepFace library to perform face detection and emotion analysis.

Dependencies
------------

*   Python 3.x
*   OpenCV (cv2)
*   shutil
*   pandas
*   DeepFace
*   firebase\_admin

Usage
-----

Before running the script, make sure that you have the required dependencies installed:

    pip install opencv-python pandas deepface firebase-admin

Also, you need to make sure that you have a service account key file named `serviceAccountKey.json` for Firebase authentication.

To execute the script, simply run the Python script in your terminal or command prompt.

    python manin.py

How It Works
------------

1.  Initializes connection with Firebase using a service account key.
2.  Lists all files in the Firebase storage bucket and downloads the latest file based on the timestamp.
3.  Uses OpenCV to capture video frames.
4.  For each frame, it extracts faces using DeepFace.
5.  Analyzes the emotions of the detected faces.
6.  Stores and prints the dominant emotion for each frame.
7.  Prints a message if the dominant emotion changes.
8.  Releases the video capture object and closes any OpenCV windows.

Notes
-----

*   The script only takes the top half of each frame for emotion detection.
*   If running the script with a GUI, a window showing the detected face will be displayed. Press 'q' to quit.

License
-------

This project is licensed under the terms of the MIT license.