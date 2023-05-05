# Emotion Detection in Video using DeepFace

This is a Python script that analyzes the emotions in a video using the DeepFace library and detects changes in the dominant emotion over time.

## Requirements

- Python 3.x
- OpenCV
- DeepFace

You can install OpenCV and DeepFace using pip:

```
pip install opencv-python
pip install deepface
```


## Usage

1. Clone the repository or download the script.
2. Install the requirements.
3. Replace "video.mp4" with the path to your video file.
4. Run the script.

The script will output a list of emotion changes in the video, including the start and end time of each change and the dominant emotion during that period.

## Parameters

You can adjust the following parameters to customize the script's behavior:

- `emotion_duration_threshold`: The minimum duration (in seconds) for an emotion to be considered a distinct period. Default is 0.5 seconds.
- `actions`: A list of DeepFace actions to perform on the video frame. Default is ['emotion'].

## Contributing

All contributors are welcome.

### For GitHub

For changes, please open an issue first to discuss what you would like to change.

### For sourcehut

Prepare a [patchset](https://man.sr.ht/git.sr.ht/#2-preparing-the-patchset).

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
