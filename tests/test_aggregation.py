import sys
from pathlib import Path
from unittest.mock import MagicMock

# Avoid loading heavy deps just to test pure logic.
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("deepface", MagicMock())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import build_segments, dominant_per_second, write_csv  # noqa: E402


def test_dominant_uses_highest_average_confidence():
    data = {0: {"happy": [90, 85], "sad": [99]}, 1: {"sad": [95]}}
    assert dominant_per_second(data) == {0: "sad", 1: "sad"}


def test_seconds_sorted():
    assert list(dominant_per_second({3: {"a": [1]}, 1: {"b": [1]}})) == [1, 3]


def test_segments_basic():
    tl = {0: "happy", 1: "happy", 2: "sad", 3: "sad"}
    assert build_segments(tl, 1) == [(0, 1, "happy"), (2, 3, "sad")]


def test_flicker_absorbed_and_remerged():
    tl = {0: "happy", 1: "happy", 2: "sad", 3: "happy", 4: "happy"}
    assert build_segments(tl, 2) == [(0, 4, "happy")]


def test_first_short_run_kept():
    tl = {0: "sad", 1: "happy", 2: "happy"}
    assert build_segments(tl, 2) == [(0, 0, "sad"), (1, 2, "happy")]


def test_empty():
    assert build_segments({}) == []


def test_write_csv(tmp_path):
    out = tmp_path / "o.csv"
    write_csv([(0, 2, "happy")], out)
    assert out.read_text().splitlines() == [
        "start_second,end_second,emotion",
        "0,2,happy",
    ]
