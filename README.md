# Gaze Bot (Python) — webcam eye-tracking avatar

Pure-Python version of the same idea: a black circular bot whose eyes slide
left/right with your gaze and blink when you blink, opened in its own
desktop window (no browser). This is a straight port of the HTML/JS project
— same tracking math, same avatar look — just running as a local Python app.

## How it works, in one paragraph

OpenCV grabs frames from your webcam. Each frame is passed to Google's
open-source **MediaPipe Face Landmarker** model, which runs locally on your
machine (nothing is uploaded) and returns 478 facial landmark points
(including iris position) plus "blendshape" expression scores. We use the
iris landmarks to work out how far left/right your pupils have moved inside
your eyes, and the `eyeBlinkLeft` / `eyeBlinkRight` blendshape scores to
detect blinking. Those three numbers drive a small hand-drawn avatar,
rendered with OpenCV drawing calls and shown in a live window.

## Project structure

```
gaze-bot-python/
├── main.py              Entry point: opens camera + window, runs the loop
├── face_tracker.py       Webcam + MediaPipe FaceLandmarker -> {gaze_x, blink_left, blink_right}
├── avatar.py              OpenCV/numpy renderer: draws the bot, smooths & animates it
├── model_downloader.py    Fetches the face-landmark model file on first run
├── requirements.txt
├── models/                Downloaded model lands here (created automatically)
└── README.md              This file
```

- **`face_tracker.py`** is the only file that knows about MediaPipe/OpenCV
  capture. It exposes `read_samples()`, returning one gaze/blink sample per
  detected face, ordered left to right.
- **`avatar.py`** knows nothing about webcams. It exposes `set_target(sample)`
  and `render()`, moving its simple eye slits with quick gaze shifts and brief
  settling, with occasional idle blinks if no face is detected.
- **`main.py`** creates one bot per detected face and arranges them in a grid.
  Click **DEBUG OFF** in the top bar to inspect the live camera feed with face,
  eye, and iris landmarks; click again to return to the bots.

## Setup (with a virtual environment)

Requires Python 3.9–3.12 (MediaPipe doesn't yet support 3.13 at the time of
writing — if `pip install mediapipe` fails, this is the first thing to check
with `python3 --version`).

**macOS / Linux:**

```bash
cd gaze-bot-python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Windows (PowerShell):**

```powershell
cd gaze-bot-python
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The first run downloads the face-landmark model (a few MB) into `models/`
automatically — you'll see a progress percentage in the terminal. After
that it's instant.

A window titled **"Gaze Bot"** opens showing the avatar. Press **`q`** or
**Esc** in that window to quit, or just close it.

### Useful flags

```bash
python main.py --preview          # also open a small window with the raw camera feed
python main.py --camera 1         # use a different webcam (if you have more than one)
python main.py --size 640         # bigger avatar window
python main.py --max-faces 12    # track more people at once
```

## Tuning it

Open `face_tracker.py` and look at the attributes set in `FaceTracker.__init__`:

| Setting       | What it does                                                                                                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dead_zone`   | How much eye movement near dead-center is ignored, to stop jitter when you're looking "straight ahead". Increase if the bot twitches when it shouldn't.                          |
| `sensitivity` | How far your iris has to travel before the bot's eyes hit full left/right. Decrease to make the bot more responsive to small eye movements; increase if it maxes out too easily. |
| `mirror_gaze` | If the bot's eyes move opposite to your own, flip this boolean.                                                                                                                  |

In `avatar.py`, `gaze_speed` in `Avatar.render()` controls how quickly the
slits move on large gaze changes and settle on small ones. The `self.smoothing`
dict controls blink speed.

The bot's look (circle size, eye size/spacing, colors) lives in
`Avatar._draw()` / `Avatar._draw_eye()` — no build step, just edit and
rerun.

## Troubleshooting

- **`pip install mediapipe` fails** — check `python3 --version`; MediaPipe
  needs 3.9–3.12. If you have a newer Python installed system-wide, create
  the venv with an older interpreter, e.g. `python3.11 -m venv .venv`.
- **Camera permission / black window** —
  - **macOS**: System Settings → Privacy & Security → Camera → allow your
    terminal app, then restart the terminal.
  - **Windows**: Settings → Privacy → Camera → allow desktop apps.
  - **Linux**: your user may need the `video` group: `sudo usermod -a -G video $USER`,
    then log out/in.
- **`Could not open camera index 0`** — try `--camera 1` (or 2...); some
  laptops enumerate a virtual camera at index 0. Also make sure no other
  app (Zoom, Teams, browser tab) is holding the camera open.
- **Bot doesn't react much** — sit in decent lighting facing the camera,
  then lower `sensitivity` in `face_tracker.py`.
- **Bot moves the wrong direction** — flip `mirror_gaze` in `face_tracker.py`.
- **Model download fails** (e.g. restrictive network) — `model_downloader.py`
  prints the direct URL; download it by hand and save it as
  `models/face_landmarker.task`.

## Credits / dependencies

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) (Apache-2.0) — face/iris landmarks + blink blendshapes, runs fully on-device.
- [OpenCV](https://opencv.org/) (Apache-2.0) — camera capture and all drawing.
- [NumPy](https://numpy.org/) (BSD) — image buffer / masking math.
