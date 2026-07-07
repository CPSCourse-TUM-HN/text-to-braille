#!/usr/bin/env python3
"""
Visualizes the current solenoid state (from braille_controller.viz_state) as
a 2x3 Braille dot grid, served as an MJPEG stream — for checking output
without the physical hardware wired up. Either import start_visualization_
server() and call it alongside the main capture script, or run this file
standalone and drive braille_controller from another process/REPL.
"""

import time
import threading
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from braille_controller import viz_lock, viz_state

CANVAS_W, CANVAS_H = 260, 340
DOT_RADIUS = 28
ON_COLOR  = (60, 200, 60)
OFF_COLOR = (60, 60, 60)
BORDER    = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Layout matches physical Braille cell numbering: 1/4 top, 2/5 middle, 3/6 bottom
DOT_POSITIONS = {
    1: (85,  90),  4: (175, 90),
    2: (85,  180), 5: (175, 180),
    3: (85,  270), 6: (175, 270),
}


def render_frame():
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 25, np.uint8)
    with viz_lock:
        dots = dict(viz_state["dots"])
        char = viz_state["char"]

    for dot, (x, y) in DOT_POSITIONS.items():
        color = ON_COLOR if dots.get(dot) else OFF_COLOR
        cv2.circle(canvas, (x, y), DOT_RADIUS, color, -1)
        cv2.circle(canvas, (x, y), DOT_RADIUS, BORDER, 2)
        cv2.putText(canvas, str(dot), (x - 6, y + 6), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    label = char.upper() if char else "(idle)"
    cv2.putText(canvas, f"char: {label}", (20, 320), FONT, 0.6,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


class VizStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path != '/':
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                frame = render_frame()
                rc, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if rc:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(enc)))
                    self.end_headers()
                    self.wfile.write(enc.tobytes()); self.wfile.write(b'\r\n')
                time.sleep(0.15)
        except Exception:
            pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True


def start_visualization_server(port=8001):
    srv = ThreadedHTTPServer(('0.0.0.0', port), VizStreamHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[STREAM] Braille cell visualization at http://<your-pi-ip>:{port}")


if __name__ == "__main__":
    start_visualization_server()
    print("Visualizer running standalone. Drive braille_controller.BrailleController "
          "from another process/REPL to see this update.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
