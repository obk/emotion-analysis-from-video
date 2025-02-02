# Video Emotion Analysis Toolkit

![OpenCV](https://img.shields.io/badge/OpenCV-5.0-%235C3EE8?logo=opencv)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79-%2300A67E)

A robust video emotion analysis system combining OpenCV's Haar cascades with DeepFace's deep learning models for efficient facial emotion recognition in video streams.

## Key Features

- **Hybrid Face Detection**: Combines OpenCV Haar cascades with DeepFace verification
- **Adaptive Frame Processing**: Skips frames (configurable) to optimize performance
- **Face Preprocessing Pipeline**: Automatic contrast/brightness adjustment + resizing
- **Confidence-based Filtering**: Ignores low-confidence predictions (<0.8 threshold)
- **Temporal Aggregation**: Groups results by second for stable emotion reporting
- **Error Resilience**: Comprehensive exception handling at all processing stages

## Installation

```bash
pip install opencv-python-headless deepface numpy
```

## Usage

```bash
python app.py <video_path>
```

**Example Analysis**:
```bash
python app.py demo_video.mp4

Frame: 15 Emotion: happy (Confidence: 0.92)
Frame: 30 Emotion: neutral (Confidence: 0.85)
Second 1: Emotion changed from happy to neutral
```

## Technical Implementation

### Processing Pipeline
1. Frame Decoding (OpenCV VideoCapture)
2. Hybrid Face Detection (Haar Cascade + DeepFace verification)
3. Face Normalization (Contrast/Brightness adjustment + Resize to 224x224)
4. Emotion Analysis (DeepFace's emotion model)
5. Temporal Aggregation (Per-second emotion statistics)

### Configuration Constants
```python
FRAME_SKIP = 5               # Process every 5th frame
EMOTION_CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence score
EMOTION_CHANGE_THRESHOLD = 0.3      # Relative change for emotion shift
```

## Optimizations

- **Selective Frame Processing**: Reduces redundant computations
- **Face Preprocessing**: Standardizes input for better model accuracy
- **Memory Management**: Explicit video capture release post-processing
- **Confidence Filtering**: Eliminates uncertain predictions

## License

MIT License - See [LICENSE](LICENSE) for full text.

## Contribution

Issues and PRs welcome! Please follow standard GitHub workflows.
