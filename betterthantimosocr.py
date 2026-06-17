#!/usr/bin/env python3
"""
Handheld Line Scanner — PaddleOCR prototype
Moves device left-to-right over text, assembles one coherent line.

Requirements:
    pip install paddlepaddle paddleocr opencv-python numpy

Usage:
    python scanner.py              # uses default camera (index 0)
    python scanner.py --camera 1   # use a different camera
    python scanner.py --demo       # run with a demo image instead of camera
"""

import cv2
import numpy as np
import argparse
import time
import sys
from collections import deque

# PaddleOCR import with helpful error
try:
    from paddleocr import PaddleOCR
except ImportError:
    print("PaddleOCR not installed. Run:")
    print("  pip install paddlepaddle paddleocr opencv-python numpy")
    sys.exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────

CAMERA_INDEX    = 0       # change if Pi camera isn't index 0
FRAME_SKIP_NORM = 12      # min pixel norm diff to consider frame "new"
MAX_LINE_GAP    = 1.5     # seconds of no new words before line is finalised
OVERLAP_THRESH  = 0.6     # word similarity threshold for dedup (0–1)
DOMINANT_RATIO  = 0.55    # a line must have >= this fraction of total text area

# Visual overlay
SHOW_DEBUG      = True    # draw bounding boxes and overlay on preview window
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ─── OCR engine (initialised once) ────────────────────────────────────────────

def init_ocr():
    """Initialise PaddleOCR — det=True so it finds text lines spatially."""
    print("Loading PaddleOCR … (first run downloads models, ~100 MB)")
    ocr = PaddleOCR(
        use_angle_cls=True,   # handles slight rotation
        lang="en",
        use_gpu=False,
        show_log=False,
        # Lite det/rec models — faster on Pi 5
        det_model_dir=None,   # uses default; swap for PP-OCRv3-mobile for speed
        rec_model_dir=None,
    )
    print("PaddleOCR ready.")
    return ocr


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(frame):
    """
    Enhance image for OCR:
    1. Greyscale
    2. CLAHE  — corrects uneven lighting / shadows
    3. Adaptive threshold — binarises without being sensitive to global brightness
    Returns a clean binary image (still 3-ch for PaddleOCR compatibility).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE: local contrast enhancement, very effective for shadows
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Mild sharpening — helps when text edge is slightly soft
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Adaptive threshold — handles lighting gradients across the page
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10
    )

    # Back to 3-channel so PaddleOCR accepts it
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ─── Dominant line selection ───────────────────────────────────────────────────

def dominant_line(ocr_result, frame_h):
    """
    From a list of detected text boxes, pick the line that:
    - is closest to vertical centre of frame
    - has the largest combined bounding-box area (most text)

    Returns list of (word, confidence, bbox) for the dominant line,
    or [] if nothing found.

    ocr_result format from PaddleOCR:
      [ [ [box_points], (text, confidence) ], ... ]
    """
    if not ocr_result or not ocr_result[0]:
        return []

    boxes = ocr_result[0]  # list of detections for this image

    # Group boxes into lines by their vertical centre (y_mid)
    # Two boxes are on the same line if their y_mid differs by < half text height
    lines = []  # list of lists of boxes

    def y_mid(box):
        pts = np.array(box[0])
        return float(pts[:, 1].mean())

    def box_height(box):
        pts = np.array(box[0])
        return float(pts[:, 1].max() - pts[:, 1].min())

    sorted_boxes = sorted(boxes, key=y_mid)

    for b in sorted_boxes:
        placed = False
        bh = box_height(b)
        for line in lines:
            rep = line[0]
            if abs(y_mid(b) - y_mid(rep)) < max(box_height(rep), bh) * 0.8:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])

    if not lines:
        return []

    # Score each line: area * centrality
    centre_y = frame_h / 2.0

    def line_score(line):
        total_area = 0
        avg_y = 0
        for b in line:
            pts = np.array(b[0])
            w = float(pts[:, 0].max() - pts[:, 0].min())
            h = float(pts[:, 1].max() - pts[:, 1].min())
            total_area += w * h
            avg_y += y_mid(b)
        avg_y /= len(line)
        # penalise distance from frame centre
        centrality = 1.0 - abs(avg_y - centre_y) / centre_y
        return total_area * (0.5 + 0.5 * centrality)

    best_line = max(lines, key=line_score)

    # Sort words left-to-right by x centre
    best_line.sort(key=lambda b: np.array(b[0])[:, 0].mean())

    # Return as (word, confidence, bbox) tuples
    result = []
    for b in best_line:
        pts = b[0]
        text, conf = b[1]
        result.append((text.strip(), conf, pts))

    return result


# ─── Sliding-window deduplication ─────────────────────────────────────────────

def normalise(word):
    return word.lower().strip(".,!?;:\"'()-")


def words_overlap(a, b):
    """True if two words are likely the same after normalisation."""
    na, nb = normalise(a), normalise(b)
    if na == nb:
        return True
    # allow for OCR noise: one char different for short words
    if len(na) <= 3 or len(nb) <= 3:
        return na == nb
    # simple character overlap ratio
    longer = max(len(na), len(nb))
    matches = sum(c1 == c2 for c1, c2 in zip(na, nb))
    return matches / longer >= OVERLAP_THRESH


def merge_into_line(assembled: list, new_words: list) -> list:
    """
    Sliding-window merge:
    Find the longest suffix of `assembled` that matches a prefix of `new_words`,
    then append only the non-overlapping tail of `new_words`.

    assembled : list of words already in the line
    new_words : words from the current frame (left-to-right order)
    returns   : updated assembled list
    """
    if not new_words:
        return assembled

    if not assembled:
        return list(new_words)

    # Try to find overlap: look for new_words[0] somewhere in the tail of assembled
    # Search window: last min(len(assembled), len(new_words)+3) words
    window = min(len(assembled), len(new_words) + 4)
    tail = assembled[-window:]

    best_overlap = 0
    best_pos = -1  # position in tail where overlap starts

    for start in range(len(tail)):
        overlap_len = 0
        for i, nw in enumerate(new_words):
            if start + i < len(tail) and words_overlap(tail[start + i], nw):
                overlap_len += 1
            else:
                break
        if overlap_len > best_overlap:
            best_overlap = overlap_len
            best_pos = start

    if best_overlap >= 1:
        # Trim assembled to the overlap point, then append non-overlapping new words
        keep_assembled = assembled[: len(assembled) - window + best_pos]
        # Take the overlapping part from new_words (better confidence potentially)
        overlap_part = new_words[:best_overlap]
        new_part = new_words[best_overlap:]
        return keep_assembled + overlap_part + new_part
    else:
        # No overlap found — just append (device may have jumped forward)
        return assembled + new_words


# ─── Frame change detection ────────────────────────────────────────────────────

def frame_changed(prev, curr, threshold=FRAME_SKIP_NORM):
    if prev is None:
        return True
    diff = cv2.absdiff(prev, curr)
    return float(diff.mean()) > threshold


# ─── Visual overlay ────────────────────────────────────────────────────────────

def draw_overlay(frame, dominant_words, assembled_line, fps):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Draw bounding boxes for dominant line words
    for word, conf, pts in dominant_words:
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(overlay, [pts_arr], isClosed=True, color=(0, 220, 80), thickness=2)
        cv2.putText(overlay, f"{word} ({conf:.2f})",
                    (pts_arr[0][0], pts_arr[0][1] - 6),
                    FONT, 0.45, (0, 220, 80), 1, cv2.LINE_AA)

    # Assembled line at bottom
    line_text = " ".join(assembled_line)
    if len(line_text) > 60:
        line_text = "…" + line_text[-57:]
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.putText(overlay, line_text, (8, h - 16),
                FONT, 0.55, (255, 255, 100), 1, cv2.LINE_AA)

    # FPS
    cv2.putText(overlay, f"{fps:.0f} fps", (w - 80, 22),
                FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ─── Demo mode ────────────────────────────────────────────────────────────────

def make_demo_frames():
    """Generate synthetic frames simulating left-to-right scan."""
    text = "The quick brown fox jumps over the lazy dog near the riverbank"
    words = text.split()
    frames = []

    full_w, h = 900, 120
    # Draw all words on a wide canvas, then slide a window across it
    canvas = np.ones((h, full_w, 3), dtype=np.uint8) * 240
    x = 10
    positions = []
    for word in words:
        sz, _ = cv2.getTextSize(word, FONT, 1.2, 2)
        positions.append((x, sz))
        cv2.putText(canvas, word, (x, 80), FONT, 1.2, (10, 10, 10), 2, cv2.LINE_AA)
        x += sz[0] + 18

    win_w = 320
    step = 40
    for start_x in range(0, full_w - win_w + 1, step):
        crop = canvas[:, start_x:start_x + win_w].copy()
        frames.append(crop)

    return frames


# ─── Main loop ────────────────────────────────────────────────────────────────

def run(camera_index=0, demo=False):
    ocr = init_ocr()

    assembled_line = []
    prev_frame = None
    last_update = time.time()
    fps_history = deque(maxlen=10)
    finalised_lines = []

    print("\n═══════════════════════════════════════")
    print("  Scanner ready. Move device left → right.")
    print("  Press  [c]  to clear current line")
    print("  Press  [s]  to save / finalise line")
    print("  Press  [q]  to quit")
    print("═══════════════════════════════════════\n")

    if demo:
        demo_frames = make_demo_frames()
        demo_idx = 0

    cap = None if demo else cv2.VideoCapture(camera_index)
    if cap and not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        sys.exit(1)

    while True:
        t0 = time.time()

        # ── Grab frame ──
        if demo:
            if demo_idx >= len(demo_frames):
                print("\nDemo complete.")
                break
            frame = demo_frames[demo_idx]
            demo_idx += 1
            time.sleep(0.08)  # simulate 12fps scan
        else:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

        # ── Skip if frame hasn't changed ──
        if not frame_changed(prev_frame, frame):
            if SHOW_DEBUG:
                display = frame.copy()
                draw_overlay(display, [], assembled_line,
                             fps_history[-1] if fps_history else 0)
                cv2.imshow("Scanner", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        prev_frame = frame.copy()

        # ── Preprocess ──
        processed = preprocess(frame)

        # ── OCR ──
        try:
            result = ocr.ocr(processed, cls=True)
        except Exception as e:
            print(f"OCR error: {e}")
            continue

        # ── Dominant line ──
        dom_words = dominant_line(result, frame.shape[0])

        if dom_words:
            word_strings = [w for w, _, _ in dom_words]
            assembled_line = merge_into_line(assembled_line, word_strings)
            last_update = time.time()
            print(f"\r  → {' '.join(assembled_line):<80}", end="", flush=True)

        # ── Auto-finalise if no new words for MAX_LINE_GAP seconds ──
        if assembled_line and (time.time() - last_update) > MAX_LINE_GAP:
            line = " ".join(assembled_line)
            finalised_lines.append(line)
            print(f"\n\n[FINALISED] {line}\n")
            assembled_line = []

        # ── FPS ──
        elapsed = time.time() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        # ── Display ──
        if SHOW_DEBUG:
            display = frame.copy()
            draw_overlay(display, dom_words, assembled_line, avg_fps)
            cv2.imshow("Scanner", display)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(f"\n[CLEARED]")
            assembled_line = []
        elif key == ord('s'):
            if assembled_line:
                line = " ".join(assembled_line)
                finalised_lines.append(line)
                print(f"\n[SAVED] {line}\n")
                assembled_line = []

    # ── Cleanup ──
    if cap:
        cap.release()
    cv2.destroyAllWindows()

    if assembled_line:
        finalised_lines.append(" ".join(assembled_line))

    print("\n\n═══════════════ FINAL OUTPUT ═══════════════")
    for i, line in enumerate(finalised_lines, 1):
        print(f"  {i}: {line}")
    print("═════════════════════════════════════════════\n")

    return finalised_lines


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handheld OCR line scanner")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help="Camera device index (default: 0)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with synthetic frames instead of camera")
    args = parser.parse_args()

    run(camera_index=args.camera, demo=args.demo)