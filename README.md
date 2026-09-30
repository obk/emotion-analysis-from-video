# Video Emotion Analysis Toolkit

![OpenCV](https://img.shields.io/badge/OpenCV-5.0-%235C3EE8?logo=opencv)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79-%2300A67E)

Detects faces in a video with OpenCV Haar cascades, classifies emotions with DeepFace, and reports how the dominant emotion changes over time.

## Key Features

- **Fast face detection**: Haar cascade (loaded once), analyzing every Nth frame and only decoding the frames it needs
- **Face preprocessing**: contrast/brightness adjustment + resize to 224x224
- **Confidence filtering**: ignores predictions below a threshold (default 80%)
- **Temporal smoothing**: reports emotion *segments*; brief flickers shorter than `--min-duration` seconds are absorbed
- **CSV export** of the segments

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py video.mp4 [--frame-skip 5] [--threshold 80] [--min-duration 2] [--csv out.csv]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--frame-skip` | 5 | analyze every Nth frame |
| `--threshold` | 80 | minimum confidence (0-100) |
| `--min-duration` | 2 | seconds an emotion must persist to count |
| `--csv` | – | write segments to a CSV file |

Example output:

```
Frame: 15 Emotion: happy (Confidence: 92.10)
Frame: 30 Emotion: neutral (Confidence: 85.40)
    0s -     4s  happy
    5s -    12s  neutral
```

Programmatic use: `run_analysis(path)` returns `[(start, end, emotion), ...]`.

## Pipeline

1. Frame decoding (OpenCV `grab`/`retrieve`, decoding only analyzed frames)
2. Face detection (Haar cascade)
3. Face normalization
4. Emotion analysis (DeepFace)
5. Per-second aggregation (highest average confidence wins)
6. Smoothing into segments

## Tests

```bash
pip install pytest && pytest
```

## License

MIT License - See [LICENSE](LICENSE) for full text.