#!/usr/bin/env python3
"""
Capture -> Gemini -> Braille pipeline.

Press ENTER, the camera records a short clip, the clip is sent inline to a
Gemini Pro model together with full context about this assistive device, and
Gemini returns a structured (Pydantic-validated) decision: the text to play
back on the Braille cell, plus guidance for the user when the footage isn't
readable. The returned text is fed to BrailleController (which mocks GPIO
when run off the Pi).

Why video instead of a photo: the user is blind and can't frame or hold the
camera perfectly — across ~3 seconds of footage Gemini can pick the sharpest,
best-framed moment on its own. Gemini samples video at 1 fps internally, so
we record a modest-fps clip; high capture fps would only inflate upload size.

Requirements:
    pip install google-genai pydantic opencv-python
    export GEMINI_API_KEY=...          (https://aistudio.google.com/apikey)

Camera: Picamera2 on the Raspberry Pi, OpenCV webcam fallback elsewhere
(so the whole flow is testable on a laptop).
"""

import os
import sys
import time
import tempfile
from enum import Enum

import cv2
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from braille_controller import BrailleController

# ─── Configuration ────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-3.1-pro-preview"
CLIP_SECONDS       = 3.0
CAPTURE_FPS        = 10          # clip framerate; Gemini samples ~1 fps anyway
FRAME_SIZE         = (1280, 720)
BRAILLE_CHAR_DELAY = 0.3         # seconds each character stays raised

# ─── What Gemini must return ──────────────────────────────────────────────────
class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class BrailleReadout(BaseModel):
    """Structured decision from Gemini about one captured clip."""

    readable: bool = Field(
        description="True only if text was confidently read and braille_text "
                    "is worth playing back to the user."
    )
    raw_text: str = Field(
        description="The text exactly as it appears in the video (the single "
                    "passage the user is aiming at). Empty if none."
    )
    braille_text: str = Field(
        description="raw_text converted for the 6-dot Braille display: only "
                    "lowercase letters a-z and spaces. Numbers spelled out as "
                    "words, punctuation dropped, abbreviations kept as "
                    "letters. Empty if not readable."
    )
    confidence: Confidence = Field(
        description="How certain the reading is. Use 'low' when guessing."
    )
    guidance: str = Field(
        description="One short spoken-style sentence of advice for the blind "
                    "user, e.g. 'Move the camera back a little, the line is "
                    "cut off on the right.' Empty if the capture was fine."
    )


SYSTEM_PROMPT = """\
You are the reading brain of an assistive device for a blind person.

The device: a Raspberry Pi with a camera and a single 6-dot Braille cell made
of solenoids. The user points the camera at printed text (a sign, a label, a
line in a book) and presses a button. You receive a short video clip from
that camera. Whatever text you return is played back on the Braille cell ONE
CHARACTER AT A TIME, each raised for about a second — so reading is slow and
every extra character costs the user real time and attention.

Your job, in order:
1. Work out what single piece of text the user is trying to read. The clip
   may contain several text fragments; pick the one that is central,
   dominant, or clearly being aimed at. Ignore background clutter.
2. Use the whole clip: the camera is handheld by a blind person, so some
   frames will be blurry, tilted, or badly framed — read from the best
   moments, and combine frames if the text is only partially visible in each.
3. Return the text adapted for the Braille cell (field braille_text):
   ONLY lowercase letters a-z and spaces — the hardware supports nothing
   else. Spell out digits as words ("Room 12" -> "room twelve"). Drop
   punctuation. Keep it as short as faithfulness allows; never pad or expand.
4. Be honest about failure. If the text is unreadable, too small, cut off,
   or there is no text at all, set readable=false, leave the text fields
   empty, and use the guidance field to tell the user — briefly, concretely,
   and in terms of physical action ("move closer", "tilt the label toward
   the camera", "slide the camera left") — how to get a better capture.
   Never guess: a wrong reading is worse than asking for another try,
   because the user cannot see the mistake.
5. If the reading succeeded but the framing was marginal (text near the
   edge, partially out of view), still return the text AND put a short
   warning in guidance.
"""

USER_PROMPT = (
    "Here is the clip just captured by the device's camera. Decide what the "
    "user is trying to read and respond in the required structure."
)

# ─── Camera abstraction: Picamera2 on the Pi, OpenCV webcam elsewhere ─────────
class PiCamera:
    def __init__(self):
        from picamera2 import Picamera2
        from libcamera import controls
        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": FRAME_SIZE, "format": "BGR888"})
        self._cam.configure(config)
        self._cam.start()
        time.sleep(0.3)
        try:
            self._cam.set_controls({
                "AfMode":  controls.AfModeEnum.Continuous,
                "AfRange": controls.AfRangeEnum.Macro,
            })
        except Exception as e:
            print(f"[camera] AF setup warning: {e}")
        time.sleep(0.4)

    def read(self):
        return self._cam.capture_array("main")

    def close(self):
        self._cam.stop()


class WebcamCamera:
    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open webcam index {index}")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("webcam frame grab failed")
        return frame

    def close(self):
        self._cap.release()


def open_camera():
    try:
        cam = PiCamera()
        print("[camera] using Picamera2 (Raspberry Pi)")
        return cam
    except ImportError:
        cam = WebcamCamera()
        print("[camera] Picamera2 not available — using OpenCV webcam")
        return cam


def record_clip(cam, seconds=CLIP_SECONDS, fps=CAPTURE_FPS):
    """Records a short clip and returns it as MP4 bytes for inline upload."""
    path = os.path.join(tempfile.gettempdir(), "braille_capture.mp4")
    writer = None
    n_frames = max(1, int(round(seconds * fps)))
    interval = 1.0 / fps

    print(f"[camera] recording {seconds:.0f}s...", end="", flush=True)
    next_t = time.time()
    for _ in range(n_frames):
        frame = cam.read()
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open (mp4v)")
        writer.write(frame)
        next_t += interval
        delay = next_t - time.time()
        if delay > 0:
            time.sleep(delay)
    writer.release()
    print(" done.")

    with open(path, "rb") as f:
        return f.read()


# ─── Gemini call ──────────────────────────────────────────────────────────────
def ask_gemini(client, video_bytes) -> BrailleReadout:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
            USER_PROMPT,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=BrailleReadout,
            temperature=0.0,
        ),
    )
    parsed = response.parsed
    if parsed is None:  # SDK failed to parse — validate the raw text ourselves
        return BrailleReadout.model_validate_json(response.text)
    return parsed


def sanitize_for_braille(text, controller):
    """Last line of defense: strip anything the hardware map can't show, so a
    stray character from the model never stalls playback mid-word."""
    cleaned = "".join(c for c in text.lower() if c in controller.BRAILLE_ALPHABET)
    return " ".join(cleaned.split())


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set. Get a key at "
              "https://aistudio.google.com/apikey and export it.")
        sys.exit(1)

    client = genai.Client()  # picks up GEMINI_API_KEY
    controller = BrailleController()
    cam = open_camera()

    print("\n═══════════════════════════════════════")
    print(f" Model: {GEMINI_MODEL}")
    print(" Point the device at text and press ENTER to read.")
    print(" Type 'q' + ENTER to quit.")
    print("═══════════════════════════════════════\n")

    try:
        while True:
            cmd = input("[ready] ENTER to capture > ").strip().lower()
            if cmd == "q":
                break

            try:
                video_bytes = record_clip(cam)
                print(f"[gemini] sending {len(video_bytes)/1e6:.1f} MB clip...")
                t0 = time.time()
                result = ask_gemini(client, video_bytes)
                print(f"[gemini] answered in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

            if result.guidance:
                print(f"[GUIDANCE] {result.guidance}")

            if not result.readable:
                print("[TEXT] (nothing readable)")
                continue

            text = sanitize_for_braille(result.braille_text, controller)
            print(f"[TEXT] {result.raw_text!r}  (confidence: {result.confidence.value})")
            print(f"[BRAILLE] playing back: {text!r}")
            controller.display_string(text, delay=BRAILLE_CHAR_DELAY)
            controller.clear_cell()

    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] stopping...")
    finally:
        cam.close()
        controller.cleanup()


if __name__ == "__main__":
    main()
