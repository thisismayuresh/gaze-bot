"""
main.py
--------
Entry point. Opens the webcam, opens a window showing the animated bot,
and (optionally) a second window with the raw camera feed for debugging.

Run with:
    python main.py
    python main.py --preview
    python main.py --camera 1 --size 640

Press 'q' or Esc in the window to quit.
"""

import argparse
import math

import cv2
import numpy as np

from avatar import Avatar, BG_COLOR
from face_tracker import FaceTracker

WINDOW_NAME = "Gaze Bot"
PREVIEW_WINDOW_NAME = "Camera preview"
TOOLBAR_HEIGHT = 48
DEBUG_BUTTON_BOUNDS = (12, 8, 112, 40)


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam eye-tracking avatar.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0). Try 1, 2... if you have multiple cameras.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also open a small window showing the raw (mirrored) camera feed.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=480,
        help="Avatar window size in pixels (default: 480).",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=8,
        help="Maximum number of people to track at once (default: 8).",
    )
    return parser.parse_args()


def render_avatar_grid(avatars, samples, size, draw=True):
    count = len(avatars)
    columns = min(count, math.ceil(math.sqrt(count * 16 / 9)))
    rows = math.ceil(count / columns)
    tile_size = min(size, 1280 // columns, 720 // rows)
    no_face = {
        "gaze_x": 0.0,
        "blink_left": 0.0,
        "blink_right": 0.0,
        "face_found": False,
    }

    tiles = []
    for index, avatar in enumerate(avatars):
        avatar.size = tile_size
        avatar.set_target(samples[index] if index < len(samples) else no_face)
        tile = avatar.render()
        if draw:
            tiles.append(tile)

    if not draw:
        return None

    blank_tile = np.full((tile_size, tile_size, 3), BG_COLOR, dtype=np.uint8)
    grid_rows = []
    for start in range(0, count, columns):
        row = tiles[start : start + columns]
        row.extend([blank_tile] * (columns - len(row)))
        grid_rows.append(np.hstack(row))
    return np.vstack(grid_rows)


def add_toolbar(image, debug_mode, face_count):
    height, width = image.shape[:2]
    toolbar = np.full((TOOLBAR_HEIGHT, width, 3), (28, 30, 32), dtype=np.uint8)
    x1, y1, x2, y2 = DEBUG_BUTTON_BOUNDS
    button_color = (65, 125, 85) if debug_mode else (70, 75, 80)
    cv2.rectangle(toolbar, (x1, y1), (x2, y2), button_color, -1)
    label = "DEBUG ON" if debug_mode else "DEBUG OFF"
    cv2.putText(
        toolbar,
        label,
        (x1 + 9, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        toolbar,
        f"FACES {face_count}",
        (x2 + 16, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 215, 215),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((toolbar, image))


def main():
    args = parse_args()

    if args.max_faces < 1:
        raise SystemExit("--max-faces must be at least 1")

    tracker = FaceTracker(camera_index=args.camera, max_faces=args.max_faces)
    avatars = [Avatar(size=args.size)]

    print("Loading model and opening camera…")
    tracker.start()
    print("Ready. Press 'q' or Esc in the window to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.size, args.size)
    if args.preview:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

    view_state = {"debug": False}

    def handle_mouse(event, x, y, _flags, _param):
        x1, y1, x2, y2 = DEBUG_BUTTON_BOUNDS
        if event == cv2.EVENT_LBUTTONUP and x1 <= x <= x2 and y1 <= y <= y2:
            view_state["debug"] = not view_state["debug"]

    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)
    display_shape = None
    try:
        while True:
            samples, frame_bgr = tracker.read_samples()
            if samples is None:
                print("Camera read failed (disconnected?). Exiting.")
                break

            desired_count = max(1, len(samples))
            avatars = avatars[:desired_count]
            avatars.extend(
                Avatar(size=args.size) for _ in range(desired_count - len(avatars))
            )
            bot_grid = render_avatar_grid(
                avatars, samples, args.size, draw=not view_state["debug"]
            )
            if view_state["debug"]:
                debug_frame = tracker.draw_debug_frame(frame_bgr)
                debug_frame = cv2.flip(debug_frame, 1)
                debug_height, debug_width = debug_frame.shape[:2]
                scale = min(1.0, 1280 / debug_width, 720 / debug_height)
                if scale < 1.0:
                    debug_frame = cv2.resize(
                        debug_frame,
                        (int(debug_width * scale), int(debug_height * scale)),
                    )
                canvas = add_toolbar(debug_frame, True, len(samples))
            else:
                canvas = add_toolbar(bot_grid, False, len(samples))
            current_shape = canvas.shape[:2]
            if current_shape != display_shape:
                cv2.resizeWindow(WINDOW_NAME, current_shape[1], current_shape[0])
                display_shape = current_shape
            cv2.imshow(WINDOW_NAME, canvas)

            if args.preview and frame_bgr is not None:
                cv2.imshow(PREVIEW_WINDOW_NAME, cv2.flip(frame_bgr, 1))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                break

            # Window closed via the [x] button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
