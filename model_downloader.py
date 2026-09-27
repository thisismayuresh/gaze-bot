"""
model_downloader.py
--------------------
Makes sure the MediaPipe "Face Landmarker" model file exists locally,
downloading it once on first run and reusing it after that.

The model is Google's own public asset (Apache-2.0, same one used by the
MediaPipe web/JS demos), a few MB in size.
"""
import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def ensure_model(url: str = MODEL_URL, path: str = MODEL_PATH) -> str:
    """Downloads the model to `path` if it isn't already there. Returns `path`."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading face landmark model (first run only)\n  from: {url}\n  to:   {path}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            done = block_num * block_size
            pct = min(100, done * 100 // total_size)
            print(f"\r  {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, path, _progress)
        print("\n  Done.")
    except Exception as exc:  # noqa: BLE001 - surface any download problem clearly
        if os.path.exists(path):
            os.remove(path)  # don't leave a partial/corrupt file behind
        raise RuntimeError(
            f"Could not download the face model automatically ({exc}).\n"
            f"You can download it manually from:\n  {url}\n"
            f"and save it as:\n  {path}"
        ) from exc

    return path


if __name__ == "__main__":
    # Lets you pre-download the model with `python model_downloader.py`.
    ensure_model()
