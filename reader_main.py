#!/usr/bin/env python3
"""
Capture -> OCR -> Braille pipeline.

Swap OCR implementations via OCR_BACKEND_NAME below or the --backend flag —
capture, keyboard triggering, and Braille output don't change either way.
All backend-specific code lives in ocr_backend.py.

Stop with 'q' or Ctrl+C.
Live camera view: http://<pi-ip>:8000
Braille cell visualization: http://<pi-ip>:8001
"""

import cv2
import numpy as np
import time
import sys
import select
import threading
import argparse
import queue
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from ocr_backend import build_backend
from braille_controller import BrailleController
from solenoid_visualizer import start_visualization_server

try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
MAIN_SIZE  = (2028, 1520)   # full-res capture used for OCR
LORES_SIZE = (320, 240)     # live view only

TRIGGER_KEYS = (' ', '\n', '\r')
QUIT_KEY     = 'q'
CAPTURE_COOLDOWN_SECONDS = 0.5

OCR_BACKEND_NAME    = "local"   # "local" or "ocrspace" — the one line to change
BRAILLE_CHAR_DELAY  = 1.2       # seconds each character stays raised

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ─── Shared state ─────────────────────────────────────────────────────────────
state_lock      = threading.Lock()
latest_view     = None
last_result     = ""
last_warning    = ""
current_status  = "press SPACE/ENTER to capture"

trigger_event = threading.Event()
stop_event    = threading.Event()
braille_q     = queue.Queue()   # recognized lines waiting for tactile playback

# ─── Camera live-view stream ───────────────────────────────────────────────────
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
                    img = latest_view.copy() if latest_view is not None else None
                if img is not None:
                    rc, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
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

def start_camera_stream(port=8000):
    srv = ThreadedHTTPServer(('0.0.0.0', port), StreamHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[STREAM] live camera view at http://<your-pi-ip>:{port}")

# ─── Keyboard trigger ─────────────────────────────────────────────────────────
def key_listener():
    if not HAS_TERMIOS or not sys.stdin.isatty():
        print("[KEY] no interactive terminal detected — keyboard trigger disabled.")
        print("[KEY] wire an alternate trigger source to `trigger_event.set()`.")
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print(f"[KEY] press SPACE or ENTER to capture, '{QUIT_KEY}' to quit.")
        while not stop_event.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch in TRIGGER_KEYS:
                trigger_event.set()
            elif ch == QUIT_KEY:
                stop_event.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ─── Camera ───────────────────────────────────────────────────────────────────
def init_camera(idx):
    try:
        from picamera2 import Picamera2
        from libcamera import controls
    except ImportError:
        print("Picamera2 not found."); sys.exit(1)
    cam = Picamera2(camera_num=idx)
    config = cam.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "BGR888"},
        lores={"size": LORES_SIZE, "format": "YUV420"},
    )
    cam.configure(config)
    cam.start(); time.sleep(0.3)
    try:
        cam.set_controls({
            "AfMode":  controls.AfModeEnum.Continuous,
            "AfRange": controls.AfRangeEnum.Macro,
        })
    except Exception as e:
        print(f"[camera] AF setup warning: {e}")
    time.sleep(0.4)
    return cam

def lores_gray(request):
    yuv = request.make_array("lores")
    h = LORES_SIZE[1]
    return yuv[:h, :]

# ─── Braille worker ────────────────────────────────────────────────────────────
def braille_worker(controller, stop):
    """Runs display_string on its own thread so tactile playback (~1.5s per
    character, by design) never blocks the capture loop. Lines are played
    back one at a time, in the order they were recognized."""
    while not stop.is_set():
        try:
            line = braille_q.get(timeout=0.2)
        except queue.Empty:
            continue
        controller.display_string(line, delay=BRAILLE_CHAR_DELAY)
        controller.clear_cell()

# ─── Main loop ─────────────────────────────────────────────────────────────────
def run(camera_index=0, camera_port=8000, viz_port=8001, backend_name=OCR_BACKEND_NAME):
    global latest_view, last_result, last_warning, current_status

    backend = build_backend(backend_name)
    print(f"[OCR] using backend: {backend_name}")

    controller = BrailleController()
    braille_stop = threading.Event()
    braille_thread = threading.Thread(target=braille_worker,
                                       args=(controller, braille_stop), daemon=True)
    braille_thread.start()

    start_camera_stream(camera_port)
    start_visualization_server(viz_port)

    cam = init_camera(camera_index)
    threading.Thread(target=key_listener, daemon=True).start()

    fps_hist = deque(maxlen=10)
    last_capture_t = 0.0

    print("\n═══════════════════════════════════════")
    print(" Point the device at a line of text.")
    print(" Press SPACE/ENTER to capture and read it.")
    print("═══════════════════════════════════════\n")

    try:
        while not stop_event.is_set():
            t0 = time.time()
            request = cam.capture_request()
            try:
                gray = lores_gray(request)

                now = time.time()
                if trigger_event.is_set():
                    trigger_event.clear()
                    if now - last_capture_t >= CAPTURE_COOLDOWN_SECONDS:
                        last_capture_t = now
                        current_status = "recognizing..."
                        main_bgr = request.make_array("main")
                        text, edge_touched, error = backend.recognize(main_bgr)
                        with state_lock:
                            if error:
                                last_warning = f"OCR failed: {error}"
                                print(f"\n[ERROR] {error}\n")
                            elif text:
                                last_result = text
                                print(f"\n[TEXT] {text}\n")
                                braille_q.put(text)
                                last_warning = (
                                    "text may be cut off at the edge — move back or reframe"
                                    if edge_touched else ""
                                )
                            else:
                                last_warning = ""
                                print("\n[TEXT] (nothing recognized)\n")
                        current_status = "press SPACE/ENTER to capture"

                el = time.time() - t0
                fps_hist.append(1.0/el if el > 0 else 0)
                fps = sum(fps_hist)/len(fps_hist)

                view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                view = cv2.resize(view, (640, 480), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(view, (0, 0), (view.shape[1], 22), (20, 20, 20), -1)
                cv2.putText(view, f"{fps:.0f}fps  {current_status}", (6, 16),
                            FONT, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
                with state_lock:
                    tail = last_result[-60:]
                    warn = last_warning
                if tail:
                    cv2.putText(view, f"text: {tail}", (6, view.shape[0]-24), FONT,
                                0.45, (120, 255, 120), 1, cv2.LINE_AA)
                if warn:
                    cv2.putText(view, warn, (6, view.shape[0]-6), FONT,
                                0.45, (100, 140, 255), 1, cv2.LINE_AA)
                with state_lock:
                    latest_view = view

            finally:
                request.release()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] stopping...")
    finally:
        stop_event.set()
        braille_stop.set()
        braille_thread.join(timeout=2.0)
        controller.cleanup()
        cam.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Capture -> OCR -> Braille reader")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--camera-port", type=int, default=8000)
    ap.add_argument("--viz-port", type=int, default=8001)
    ap.add_argument("--backend", default=OCR_BACKEND_NAME, choices=["local", "ocrspace"])
    args = ap.parse_args()
    run(camera_index=args.camera, camera_port=args.camera_port,
        viz_port=args.viz_port, backend_name=args.backend)
