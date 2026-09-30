import argparse
import csv
from collections import defaultdict
from functools import lru_cache

import cv2
from deepface import DeepFace

FRAME_SKIP = 5  # Analyze every Nth frame to reduce noise and improve speed
# DeepFace reports emotion scores as percentages (0-100), not fractions.
EMOTION_CONFIDENCE_THRESHOLD = 80.0  # Ignore low-confidence predictions
MIN_SEGMENT_SECONDS = 2  # Emotion must last this long to count as a change
FACE_SIZE = (224, 224)  # Size expected by DeepFace
CONTRAST = 1.2  # Contrast control (1.0-3.0)
BRIGHTNESS = 30  # Brightness control (0-100)


@lru_cache(maxsize=1)
def get_face_cascade():
    """Load the Haar cascade once instead of on every frame."""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def preprocess_face(face_image):
    """Adjust brightness/contrast and resize face."""
    adjusted = cv2.convertScaleAbs(face_image, alpha=CONTRAST, beta=BRIGHTNESS)
    return cv2.resize(adjusted, FACE_SIZE, interpolation=cv2.INTER_LINEAR)


def detect_faces(frame):
    """Detect face boxes (x, y, w, h) with a Haar cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )


def dominant_per_second(emotions_per_second):
    """Pick each second's emotion with the highest average confidence."""
    return {
        sec: max(data, key=lambda e: sum(data[e]) / len(data[e]))
        for sec, data in sorted(emotions_per_second.items())
    }


def build_segments(timeline, min_duration=1):
    """Collapse {second: emotion} into [(start, end, emotion)] segments.

    Runs of the same emotion shorter than `min_duration` seconds are treated
    as flicker and absorbed into the preceding segment (the first segment is
    kept as-is so nothing is dropped from the start). Seconds without
    detections don't break a run.
    """
    runs = []  # [emotion, start, end, count]
    for sec, emotion in sorted(timeline.items()):
        if runs and runs[-1][0] == emotion:
            runs[-1][2] = sec
            runs[-1][3] += 1
        else:
            runs.append([emotion, sec, sec, 1])

    segments = []
    for emotion, start, end, count in runs:
        if segments and count < min_duration:
            segments[-1][1] = end  # absorb flicker into previous segment
        elif segments and segments[-1][2] == emotion:
            segments[-1][1] = end  # re-merge after absorbing flicker
        else:
            segments.append([start, end, emotion])
    return [tuple(seg) for seg in segments]


def print_segments(segments):
    for start, end, emotion in segments:
        print(f"{start:>5}s - {end:>5}s  {emotion}")


def write_csv(segments, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_second", "end_second", "emotion"])
        writer.writerows(segments)


def run_analysis(
    video_path,
    frame_skip=FRAME_SKIP,
    threshold=EMOTION_CONFIDENCE_THRESHOLD,
    min_duration=MIN_SEGMENT_SECONDS,
):
    """Analyze a video and return [(start, end, emotion)] segments."""
    print(f"Running analysis on {video_path}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Some containers report 0
    # {second: {emotion: [confidences]}}
    emotions_per_second = defaultdict(lambda: defaultdict(list))

    try:
        frame_count = -1
        while True:
            frame_count += 1
            # grab() advances without decoding; only decode frames we analyze
            if not cap.grab():
                break
            if frame_count % frame_skip != 0:
                continue
            ret, frame = cap.retrieve()
            if not ret:
                continue

            for x, y, w, h in detect_faces(frame):
                processed_face = preprocess_face(frame[y : y + h, x : x + w])

                try:
                    result = DeepFace.analyze(
                        processed_face, actions=["emotion"], enforce_detection=False
                    )[0]
                except Exception as e:
                    print(f"Error analyzing frame {frame_count}: {e}")
                    continue

                emotion = result["dominant_emotion"]
                confidence = result["emotion"][emotion]
                if confidence < threshold:
                    continue

                print(
                    f"Frame: {frame_count} Emotion: {emotion} "
                    f"(Confidence: {confidence:.2f})"
                )
                emotions_per_second[int(frame_count / fps)][emotion].append(confidence)
    finally:
        cap.release()

    segments = build_segments(dominant_per_second(emotions_per_second), min_duration)
    print_segments(segments)
    return segments


def main():
    parser = argparse.ArgumentParser(description="Video emotion analysis")
    parser.add_argument("video_path")
    parser.add_argument(
        "--frame-skip", type=int, default=FRAME_SKIP,
        help=f"analyze every Nth frame (default {FRAME_SKIP})",
    )
    parser.add_argument(
        "--threshold", type=float, default=EMOTION_CONFIDENCE_THRESHOLD,
        help=f"minimum confidence, 0-100 (default {EMOTION_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--min-duration", type=int, default=MIN_SEGMENT_SECONDS,
        help=f"seconds an emotion must persist to be reported (default {MIN_SEGMENT_SECONDS})",
    )
    parser.add_argument("--csv", metavar="PATH", help="write emotion segments to CSV")
    args = parser.parse_args()
    if args.frame_skip < 1:
        parser.error("--frame-skip must be >= 1")

    segments = run_analysis(
        args.video_path, args.frame_skip, args.threshold, args.min_duration
    )
    if args.csv:
        write_csv(segments, args.csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
