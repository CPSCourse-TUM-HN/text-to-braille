#!/usr/bin/env python3
"""
Mosaic Line Scanner v2 — Pi 4B / Camera Module 3 Noir
═══════════════════════════════════════════════════════════════════════════════
Changes vs v1 (driven by real captures: tilted baseline, wobble, variable speed):

  1. DESKEW the assembled strip before OCR  (minAreaRect angle -> rotate flat).
     v1 pasted tilted text into a flat strip, shearing glyphs at every seam
     -> phantom words ("Witnou Maue"). Deskew removes residual rotation.

  2. RECOGNITION-ONLY OCR on projection-profile word segments.
     v1 ran the slow detector on the whole strip, which mis-split / re-ordered
     boxes. Now: deskew -> split at whitespace gaps -> rec-only per chunk ->
     join left-to-right. Skips the detector entirely; order is positional.

  3. LINE-END is no longer guessed from pixel motion.
     A blank gap looks identical to a stop (no texture either way), and slowing
     down used to false-trigger finalization -> "only the end of the line".
     Use a GPIO button (USE_BUTTON) or just a generous idle fallback.

  4. INVERT_X default flipped to match a normal L->R scan on this rig.
     (v1's default piled every frame at x=0 -> smear. If yours smears, flip it.)

Per-frame: grayscale crop + 1 phaseCorrelate (cheap).
Per-line : deskew + word-segment + rec-only OCR (runs once, on a clean strip).

Stop with Ctrl+C.  Watch the live mosaic at http://<pi-ip>:8000.
"""

import cv2
import numpy as np
import time
import sys
import queue
import threading
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    from rapidocr import RapidOCR
except ImportError:
    print("RapidOCR missing. Run: pip install rapidocr onnxruntime opencv-python-headless numpy")
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
CAPTURE_SIZE  = (640, 480)

STRIP_TOP_RATIO = 0.20      # vertical band of the frame holding the text line
STRIP_BOT_RATIO = 0.80

VPAD          = 50          # vertical slack for wobble (px). Bigger if wobbly.
MAX_MOSAIC_W  = 6000        # finalize if a line gets this wide (px)

# Motion / coast
STILL_DIFF    = 4.0         # mean abs frame diff below this == not moving
MIN_CORR_RESP = 0.06        # phaseCorrelate confidence floor; below -> coast
MAX_STEP_PX   = 80          # per-frame pan above this == too fast (warn)

# Sign conventions. If the live mosaic builds backwards / smears, flip INVERT_X.
INVERT_X      = True
INVERT_Y      = True

# ── Line-end detection ──
# Pixel-based idle is unreliable (blank gap == stop). Prefer a hardware delimiter.
USE_BUTTON    = False       # True -> press GPIO button to end a line (robust)
BUTTON_PIN    = 17          # BCM pin, wired button -> GND, internal pull-up
IDLE_SECONDS  = 1.3         # fallback: held still this long -> end of line
                            # (generous, to bridge slow-downs & word gaps)

# OCR
OCR_THREADS   = 3           # leave 1 core free on the 4-core Pi 4B
FORCE_EN      = True       # KEEP False for German (EN model drops ä ö ü ß)
USE_REC_ONLY  = True        # deskew+segment+rec-only. False -> full det+rec.

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ─── Shared state ─────────────────────────────────────────────────────────────
state_lock         = threading.Lock()
latest_mosaic_view = None
last_finalized     = ""

# ─── Debug stream ─────────────────────────────────────────────────────────────
class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path != '/':
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with state_lock:
                    img = latest_mosaic_view.copy() if latest_mosaic_view is not None else None
                if img is not None:
                    rc, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
                    if rc:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(enc)))
                        self.end_headers()
                        self.wfile.write(enc.tobytes()); self.wfile.write(b'\r\n')
                time.sleep(0.08)
        except Exception:
            pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True

def start_stream(port=8000):
    srv = ThreadedHTTPServer(('0.0.0.0', port), StreamHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[STREAM] live mosaic at http://<your-pi-ip>:{port}")

# ─── Camera ───────────────────────────────────────────────────────────────────
def init_camera(idx):
    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except ImportError:
        print("Picamera2 not found."); sys.exit(1)
    cam = Picamera2(camera_num=idx)
    cam.configure(cam.create_video_configuration(
        main={"size": CAPTURE_SIZE, "format": "BGR888"}))
    cam.start(); time.sleep(0.3)
    try:
        w, h = CAPTURE_SIZE
        band = (0, int(h*STRIP_TOP_RATIO), w, int(h*(STRIP_BOT_RATIO-STRIP_TOP_RATIO)))
        cam.set_controls({
            "AfMode":    controls.AfModeEnum.Continuous,
            "AfRange":   controls.AfRangeEnum.Macro,
            "AfWindows": [band],
        })
    except Exception as e:
        print(f"[camera] AF setup warning: {e}")
    time.sleep(0.4)
    return cam

def get_strip(frame):
    h = frame.shape[0]
    y0, y1 = int(h*STRIP_TOP_RATIO), int(h*STRIP_BOT_RATIO)
    gray = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY)
    return gray, gray.astype(np.float32)

# ─── Mosaic accumulator ───────────────────────────────────────────────────────
class Mosaic:
    def __init__(self, strip_h, strip_w):
        self.sh, self.sw = strip_h, strip_w
        self.height = strip_h + 2*VPAD
        self.canvas = np.full((self.height, MAX_MOSAIC_W), 255, np.uint8)
        self.reset()
    def reset(self):
        self.canvas[:] = 255
        self.acc_x = self.acc_y = 0.0
        self.written = 0; self.last_pan = 0.0
        self.has_content = False
    def add(self, strip_gray, pan_x, pan_y):
        self.acc_x += pan_x; self.acc_y += pan_y; self.last_pan = pan_x
        x0 = int(round(self.acc_x))
        y0 = VPAD + int(round(self.acc_y))
        y0 = max(0, min(y0, self.height - self.sh))
        if x0 < 0: x0 = 0
        if x0 + self.sw > MAX_MOSAIC_W: return False
        self.canvas[y0:y0+self.sh, x0:x0+self.sw] = strip_gray
        self.written = max(self.written, x0+self.sw)
        self.has_content = True
        return True
    def image(self):
        return None if self.written <= 0 else self.canvas[:, :self.written]
    def tail_view(self, width=900):
        if self.written <= 0: return None
        x0 = max(0, self.written - width)
        return cv2.cvtColor(self.canvas[:, x0:self.written], cv2.COLOR_GRAY2BGR)

# ─── Deskew + word segmentation + OCR ─────────────────────────────────────────
def deskew(gray):
    """Flatten residual baseline rotation via minAreaRect on the ink mask."""
    inv = 255 - gray
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 80:
        return gray
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle >  45: angle -= 90
    if angle < -45: angle += 90
    if abs(angle) < 0.3:                      # already flat
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)

def segment_words(gray):
    """Split the strip at whitespace gaps using a vertical ink projection.
    Merging adjacent words is harmless; the goal is to never split a glyph and
    to keep each chunk within the recognizer's comfortable width."""
    inv = 255 - gray
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    has_ink = mask.sum(axis=0) > (mask.shape[0] * 255 * 0.02)
    gap_thresh = max(8, int(gray.shape[0] * 0.22))   # min whitespace to split
    segs, start, gap = [], None, 0
    for i, v in enumerate(has_ink):
        if v:
            if start is None: start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= gap_thresh:
                segs.append((max(0, start-4), i - gap + 4)); start = None
    if start is not None:
        segs.append((max(0, start-4), len(has_ink)))
    return segs

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def build_ocr():
    params = {}
    try:
        from rapidocr import ModelType, LangRec
        params["Det.model_type"] = ModelType.MOBILE
        params["Rec.model_type"] = ModelType.MOBILE
        if FORCE_EN:
            params["Rec.lang_type"] = LangRec.EN
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = OCR_THREADS
    except Exception:
        pass
    try:
        return RapidOCR(params=params) if params else RapidOCR()
    except TypeError:
        return RapidOCR()

def _rec_text(res):
    """Pull text out of a rec-only (TextRecOutput) or full result."""
    if res is None: return ""
    txts = getattr(res, "txts", None)
    if not txts: return ""
    return " ".join(t.strip() for t in txts if t and t.strip())

def ocr_line(ocr, mosaic_gray):
    img = _clahe.apply(mosaic_gray)
    img = deskew(img)
    if not USE_REC_ONLY:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            res = ocr(bgr)
        except Exception as e:
            print(f"[OCR] error: {e}"); return ""
        if res is None or getattr(res, "boxes", None) is None: return ""
        items = []
        for i in range(len(res.boxes)):
            t = (res.txts[i] or "").strip()
            if t:
                items.append((float(np.array(res.boxes[i])[:, 0].min()), t))
        items.sort(key=lambda z: z[0])
        return " ".join(t for _, t in items)

    # rec-only path: deskew -> segments -> recognize each, in order
    words = []
    for (a, b) in segment_words(img):
        chunk = cv2.cvtColor(img[:, a:b], cv2.COLOR_GRAY2BGR)
        try:
            res = ocr(chunk, use_det=False, use_cls=False, use_rec=True)
        except Exception as e:
            print(f"[OCR] rec-only error: {e}")
            continue
        txt = _rec_text(res)
        if txt:
            words.append(txt)
    return " ".join(words)

def ocr_worker(q, stop):
    global last_finalized
    print("Loading RapidOCR...")
    ocr = build_ocr()
    print("RapidOCR ready.")
    while not stop.is_set():
        try:
            mosaic_gray = q.get(timeout=0.2)
        except queue.Empty:
            continue
        line = ocr_line(ocr, mosaic_gray)
        if line.strip():
            with state_lock:
                last_finalized = line
            print(f"\n[LINE] {line}\n")
            # ── feed `line` to your Braille pipeline here ──

# ─── Optional GPIO button for line-end ────────────────────────────────────────
def setup_button():
    if not USE_BUTTON:
        return None
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"[BUTTON] line-end on GPIO{BUTTON_PIN} (press to GND)")
        return GPIO
    except Exception as e:
        print(f"[BUTTON] unavailable ({e}); falling back to idle timeout")
        return None

# ─── Main loop ────────────────────────────────────────────────────────────────
def run(camera_index=0, port=8000):
    global latest_mosaic_view

    start_stream(port)
    ocr_q  = queue.Queue(maxsize=4)
    stop   = threading.Event()
    worker = threading.Thread(target=ocr_worker, args=(ocr_q, stop), daemon=True)
    worker.start()

    gpio = setup_button()
    cam  = init_camera(camera_index)

    first = cam.capture_array()
    sh = int(first.shape[0]*STRIP_BOT_RATIO) - int(first.shape[0]*STRIP_TOP_RATIO)
    sw = first.shape[1]
    mosaic = Mosaic(sh, sw)
    win = cv2.createHanningWindow((sw, sh), cv2.CV_32F)

    prev_u8 = prev_f = None
    last_motion_t = time.time()
    btn_prev = 1
    fps_hist = deque(maxlen=10)

    print("\n═══════════════════════════════════════")
    print(" Scan a line left -> right.", "Press the button to end it."
          if gpio else "Pause at the end to read it out.")
    print(" Flip INVERT_X if the mosaic builds backwards. Ctrl+C to quit.")
    print("═══════════════════════════════════════\n")

    try:
        while True:
            t0 = time.time()
            frame = cam.capture_array()
            if frame is None: break
            gray_u8, gray_f = get_strip(frame)

            moving = False
            status = "idle"
            if prev_u8 is not None:
                if float(cv2.absdiff(prev_u8, gray_u8).mean()) >= STILL_DIFF:
                    moving = True
                    (dx, dy), resp = cv2.phaseCorrelate(prev_f, gray_f, win)
                    pan_x = -dx if INVERT_X else dx
                    pan_y = -dy if INVERT_Y else dy
                    if resp < MIN_CORR_RESP:
                        pan_x, pan_y = mosaic.last_pan, 0.0
                        status = "coast (low texture)"
                    elif abs(pan_x) > MAX_STEP_PX:
                        status = "TOO FAST — slow down"
                    else:
                        status = f"scan pan={pan_x:+.1f} conf={resp:.2f}"
                    if not mosaic.add(gray_u8, pan_x, pan_y):
                        _finalize(mosaic, ocr_q)
                    last_motion_t = time.time()

            # ── line-end ──
            end_line = False
            if gpio is not None:
                b = gpio.input(BUTTON_PIN)
                if b == 0 and btn_prev == 1:          # falling edge = press
                    end_line = True
                btn_prev = b
            elif (not moving and mosaic.has_content
                  and time.time() - last_motion_t > IDLE_SECONDS):
                end_line = True
            if end_line and mosaic.has_content:
                _finalize(mosaic, ocr_q); status = "finalized line"

            prev_u8, prev_f = gray_u8, gray_f

            el = time.time() - t0
            fps_hist.append(1.0/el if el > 0 else 0)
            fps = sum(fps_hist)/len(fps_hist)

            view = mosaic.tail_view(900)
            if view is None:
                view = np.full((sh+2*VPAD, 900, 3), 30, np.uint8)
            cv2.rectangle(view, (0, 0), (view.shape[1], 22), (20, 20, 20), -1)
            cv2.putText(view, f"{fps:.0f}fps  {status}", (6, 16), FONT, 0.45,
                        (180, 220, 255), 1, cv2.LINE_AA)
            with state_lock:
                tail = last_finalized[-50:]
            if tail:
                cv2.putText(view, f"last: {tail}", (6, view.shape[0]-8), FONT,
                            0.45, (120, 255, 120), 1, cv2.LINE_AA)
            with state_lock:
                latest_mosaic_view = view
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[INFO] stopping...")
    finally:
        if mosaic.has_content:
            _finalize(mosaic, ocr_q); time.sleep(0.5)
        stop.set(); worker.join(timeout=2.0); cam.stop()
        if gpio is not None:
            gpio.cleanup()

def _finalize(mosaic, ocr_q):
    img = mosaic.image()
    if img is not None:
        try: ocr_q.put_nowait(img.copy())
        except queue.Full: pass
    mosaic.reset()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mosaic line scanner v2 — Pi 4B")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--port",   type=int, default=8000)
    ap.add_argument("--invert-x", action="store_true", help="flip horizontal build direction")
    ap.add_argument("--button", action="store_true", help="use GPIO button for line-end")
    args = ap.parse_args()
    if args.invert_x: INVERT_X = not INVERT_X
    if args.button:   USE_BUTTON = True
    run(camera_index=args.camera, port=args.port)