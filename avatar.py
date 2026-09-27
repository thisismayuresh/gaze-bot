"""
avatar.py
----------
Python twin of the JS project's `avatar.js`: draws a simple black circular
bot with two eyes onto a numpy image using OpenCV drawing calls, and
animates its eye slits with quick gaze shifts and brief settling.

Knows nothing about webcams or MediaPipe -- it only consumes
{gaze_x, blink_left, blink_right, face_found} samples.
"""

import random
import time

import cv2
import numpy as np

BODY_COLOR = (10, 10, 10)  # BGR, near-black
EYE_WHITE = (248, 246, 245)  # BGR
BG_COLOR = (24, 20, 18)  # BGR, page-like backdrop behind the circle


class Avatar:
    def __init__(self, size: int = 480):
        self.size = size

        # Smoothed (rendered) state.
        self.state = {"gaze_x": 0.0, "blink_left": 0.0, "blink_right": 0.0}
        # Latest readings from the tracker; state eases toward this each frame.
        self.target = dict(self.state)

        # Blinks are quick; gaze uses saccades with a short settling phase.
        self.smoothing = {"blink_left": 0.6, "blink_right": 0.6}

        self.face_found = False
        self._idle_next_blink_at = time.time() + self._random_idle_interval()

    # -- public API ----------------------------------------------------

    def set_target(self, sample: dict):
        self.target["gaze_x"] = sample["gaze_x"]
        self.target["blink_left"] = sample["blink_left"]
        self.target["blink_right"] = sample["blink_right"]
        self.face_found = sample["face_found"]
        if self.face_found:
            self._idle_next_blink_at = time.time() + self._random_idle_interval()

    def render(self) -> np.ndarray:
        """Advances the animation by one step and returns a BGR image."""
        if not self.face_found:
            self._maybe_idle_blink()

        gaze_delta = self.target["gaze_x"] - self.state["gaze_x"]
        if abs(gaze_delta) < 0.015:
            gaze_delta = 0.0
        gaze_speed = 0.82 if abs(gaze_delta) >= 0.18 else 0.30
        self.state["gaze_x"] += gaze_delta * gaze_speed

        for key, factor in self.smoothing.items():
            self.state[key] += (self.target[key] - self.state[key]) * factor

        return self._draw()

    # -- idle blinking (keeps the bot feeling alive with no face detected) --

    @staticmethod
    def _random_idle_interval():
        return random.uniform(2.5, 5.5)

    def _maybe_idle_blink(self):
        if time.time() >= self._idle_next_blink_at:
            self._idle_next_blink_at = time.time() + self._random_idle_interval()
            # A quick synthetic blink pulse; render() is called every frame so
            # nudging the *target* here is enough -- smoothing animates it.
            self.target["blink_left"] = 1.0
            self.target["blink_right"] = 1.0
            self._idle_blink_release_at = time.time() + 0.09

        if (
            hasattr(self, "_idle_blink_release_at")
            and time.time() >= self._idle_blink_release_at
        ):
            self.target["blink_left"] = 0.0
            self.target["blink_right"] = 0.0
            del self._idle_blink_release_at

    # -- drawing ---------------------------------------------------------

    def _draw(self) -> np.ndarray:
        size = self.size
        canvas = np.full((size, size, 3), BG_COLOR, dtype=np.uint8)

        cx, cy = size // 2, size // 2
        radius = int(size * 0.42)

        cv2.circle(canvas, (cx, cy), radius, BODY_COLOR, -1, lineType=cv2.LINE_AA)

        eye_offset_x = radius * 0.20
        eye_y = cy - int(radius * 0.18)
        eye_half_w = radius * 0.065
        eye_half_h_max = radius * 0.18
        max_eye_shift = radius * 0.14

        gaze = self.state["gaze_x"]  # -1..1
        eye_shift = gaze * max_eye_shift

        self._draw_eye(
            canvas,
            cx - eye_offset_x,
            eye_y,
            eye_half_w,
            eye_half_h_max,
            eye_shift,
            self.state["blink_left"],
        )
        self._draw_eye(
            canvas,
            cx + eye_offset_x,
            eye_y,
            eye_half_w,
            eye_half_h_max,
            eye_shift,
            self.state["blink_right"],
        )

        return canvas

    @staticmethod
    def _draw_eye(canvas, cx, cy, half_w, half_h_max, eye_shift, blink):
        """Draw a simple upright eye slit that narrows during a blink."""
        openness = max(0.04, 1 - blink)
        axes = (max(1, int(round(half_w))), max(1, int(round(half_h_max * openness))))
        center = (int(round(cx + eye_shift)), int(round(cy)))
        cv2.ellipse(
            canvas,
            center,
            axes,
            0,
            0,
            360,
            EYE_WHITE,
            -1,
            lineType=cv2.LINE_AA,
        )
