"""
face_tracker.py
----------------
Python twin of the JS project's `faceTracker.js`: wraps the webcam and
Google's MediaPipe "Face Landmarker" task to turn each camera frame into
one gaze/blink sample for each detected face:

    gaze_x      -1 (looking left) ... 0 (center) ... +1 (looking right)
    blink_left  0 (open) ... 1 (fully closed)   -- subject's left eye
    blink_right 0 (open) ... 1 (fully closed)   -- subject's right eye

Same approach as the browser version: iris landmarks (indices 468-477,
enabled via face-mesh "refinement") give pupil position relative to the eye
corners, and ready-made "blendshape" scores (`eyeBlinkLeft` / `eyeBlinkRight`)
give a far more reliable blink signal than hand-rolled eye-aspect-ratio math.
"""

from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from model_downloader import ensure_model

# Landmark indices from MediaPipe's 478-point face mesh (with iris refinement).
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
LEFT_IRIS_CENTER = 473

RIGHT_EYE_OUTER_CORNER = 33
RIGHT_EYE_INNER_CORNER = 133
RIGHT_IRIS_CENTER = 468

FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
LEFT_EYE_CONTOUR = (
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
)
RIGHT_EYE_CONTOUR = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
)


def _map_clamped(value, in_min, in_max, out_min, out_max):
    t = (value - in_min) / (in_max - in_min)
    t = min(1.0, max(0.0, t))
    return out_min + t * (out_max - out_min)


def _blendshape_score(categories, name):
    for c in categories:
        if c.category_name == name:
            return c.score
    return 0.0


class FaceTracker:
    def __init__(
        self, camera_index: int = 0, model_path: str | None = None, max_faces: int = 8
    ):
        self.camera_index = camera_index
        self.model_path = model_path or ensure_model()
        self.max_faces = max_faces
        self.cap = None
        self.landmarker = None
        self._last_ts_ms = -1
        self._last_landmarks = []

        # Tune these if the bot feels too twitchy or not responsive enough.
        self.dead_zone = 0.06  # gaze magnitude below this is treated as "center"
        self.sensitivity = 0.32  # how far the iris must travel to hit full left/right
        self.mirror_gaze = True  # flip if the bot moves opposite to your eyes

    def start(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=self.max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Try a different --camera index, and make sure no other app is using it."
            )

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def read_samples(self):
        """
        Captures one frame and returns (samples, frame_bgr).
        Each sample contains gaze_x, blink_left, blink_right, and face_found.
        Samples are ordered from left to right in the camera frame.
        Returns (None, None) if the camera read failed (e.g. disconnected).
        """
        ok, frame_bgr = self.cap.read()
        if not ok:
            return None, None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        faces = []
        blendshape_faces = result.face_blendshapes or []
        for index, landmarks in enumerate(result.face_landmarks):
            blendshapes = (
                blendshape_faces[index] if index < len(blendshape_faces) else []
            )
            faces.append(
                (
                    sum(point.x for point in landmarks) / len(landmarks),
                    landmarks,
                    {
                        "gaze_x": self._compute_gaze_x(landmarks),
                        "blink_left": _blendshape_score(blendshapes, "eyeBlinkLeft"),
                        "blink_right": _blendshape_score(blendshapes, "eyeBlinkRight"),
                        "face_found": True,
                    },
                )
            )

        faces.sort(key=lambda item: item[0])
        self._last_landmarks = [landmarks for _, landmarks, _ in faces]
        return [sample for _, _, sample in faces], frame_bgr

    def draw_debug_frame(self, frame_bgr):
        """Overlay face landmarks, eye contours, and iris centers on a frame."""
        annotated = frame_bgr.copy()
        height, width = annotated.shape[:2]
        face_color = (80, 190, 150)
        eye_color = (40, 230, 220)
        iris_color = (40, 130, 255)

        for landmarks in self._last_landmarks:
            points = [
                (int(point.x * width), int(point.y * height)) for point in landmarks
            ]
            for x, y in points:
                cv2.circle(annotated, (x, y), 1, face_color, -1, cv2.LINE_AA)

            for contour, color, closed in (
                (FACE_OVAL, face_color, True),
                (LEFT_EYE_CONTOUR, eye_color, True),
                (RIGHT_EYE_CONTOUR, eye_color, True),
                ((168, 6, 197, 195, 5, 4, 1), face_color, False),
                ((468, 469, 470, 471, 472), iris_color, True),
                ((473, 474, 475, 476, 477), iris_color, True),
            ):
                contour_points = np.array(
                    [points[index] for index in contour], dtype=np.int32
                )
                cv2.polylines(
                    annotated, [contour_points], closed, color, 1, cv2.LINE_AA
                )

            for center_index in (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                center = points[center_index]
                cv2.circle(annotated, center, 5, iris_color, 1, cv2.LINE_AA)
                cv2.circle(annotated, center, 2, (255, 255, 255), -1, cv2.LINE_AA)

        return annotated

    def read_sample(self):
        """Compatibility wrapper returning the leftmost face sample."""
        samples, frame_bgr = self.read_samples()
        if samples is None or samples:
            return (samples[0] if samples else None), frame_bgr
        return {
            "gaze_x": 0.0,
            "blink_left": 0.0,
            "blink_right": 0.0,
            "face_found": False,
        }, frame_bgr

    def _compute_gaze_x(self, landmarks):
        right_ratio = self._iris_ratio(
            landmarks[RIGHT_EYE_OUTER_CORNER],
            landmarks[RIGHT_EYE_INNER_CORNER],
            landmarks[RIGHT_IRIS_CENTER],
        )
        left_ratio = self._iris_ratio(
            landmarks[LEFT_EYE_INNER_CORNER],
            landmarks[LEFT_EYE_OUTER_CORNER],
            landmarks[LEFT_IRIS_CENTER],
        )

        avg_ratio = (right_ratio + left_ratio) / 2
        gaze = (avg_ratio - 0.5) * 2  # -> roughly -1..1

        if abs(gaze) < self.dead_zone:
            gaze = 0.0

        gaze = _map_clamped(gaze, -self.sensitivity, self.sensitivity, -1.0, 1.0)

        return -gaze if self.mirror_gaze else gaze

    @staticmethod
    def _iris_ratio(corner_a, corner_b, iris):
        span = corner_b.x - corner_a.x
        if abs(span) < 1e-6:
            return 0.5
        return (iris.x - corner_a.x) / span
