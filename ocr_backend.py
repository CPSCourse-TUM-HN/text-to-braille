#!/usr/bin/env python3
"""
OCR backend abstraction — lets the capture loop swap OCR implementations
(local RapidOCR, OCR.space API, or anything added later) without touching
capture, triggering, or Braille-output code. Every backend exposes the same
method:

    text, edge_touched, error = backend.recognize(bgr_frame)

  text         - recognized text (collapsed to a single line), "" if none
  edge_touched - True if a recognized word's box touches the frame edge
                 (likely truncated — caller should warn the user to reframe)
  error        - non-empty string on failure; caller should not trust `text`
                 when this is set

To add a new backend: subclass OCRBackend, implement recognize(), register
it in build_backend() below. Nothing else in the pipeline needs to change.
"""

from abc import ABC, abstractmethod
import os
import cv2
import numpy as np

EDGE_MARGIN_PX = 20  # shared definition of "touching the edge" across backends


class OCRBackend(ABC):
    @abstractmethod
    def recognize(self, bgr_frame):
        """Returns (text: str, edge_touched: bool, error: str)."""
        ...


class LocalRapidOCRBackend(OCRBackend):
    """Runs RapidOCR locally on the Pi. No network dependency, no per-call
    cost, but bounded by on-device model quality and CPU speed."""

    def __init__(self, force_en=True, threads=3):
        from rapidocr import RapidOCR
        params = {}
        try:
            from rapidocr import ModelType, LangRec
            params["Det.model_type"] = ModelType.MOBILE
            params["Rec.model_type"] = ModelType.MOBILE
            if force_en:
                params["Rec.lang_type"] = LangRec.EN
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
        except Exception:
            pass
        try:
            self._ocr = RapidOCR(params=params) if params else RapidOCR()
        except TypeError:
            self._ocr = RapidOCR()

    def recognize(self, bgr_frame):
        frame_w = bgr_frame.shape[1]
        try:
            res = self._ocr(bgr_frame)
        except Exception as e:
            return "", False, f"local OCR error: {e}"
        if res is None or getattr(res, "boxes", None) is None:
            return "", False, ""

        items = []
        edge_touched = False
        for i in range(len(res.boxes)):
            t = (res.txts[i] or "").strip()
            if not t:
                continue
            xs = np.array(res.boxes[i])[:, 0]
            x_min, x_max = float(xs.min()), float(xs.max())
            if x_min <= EDGE_MARGIN_PX or x_max >= frame_w - EDGE_MARGIN_PX:
                edge_touched = True
            items.append((x_min, t))
        items.sort(key=lambda z: z[0])
        return " ".join(t for _, t in items), edge_touched, ""


class OCRSpaceBackend(OCRBackend):
    """Uploads the frame to the OCR.space API (https://ocr.space/ocrapi).
    Needs internet access and an API key; adds network latency; subject to
    free-tier rate/size limits."""

    ENDPOINT = "https://api.ocr.space/parse/image"

    def __init__(self, api_key=None, language="eng", engine="2",
                 timeout=15, jpeg_quality=85):
        import requests
        self._requests = requests
        self.api_key = api_key or os.environ.get("OCR_SPACE_API_KEY", "")
        self.language = language
        self.engine = engine
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        if not self.api_key:
            print("[OCRSpaceBackend] WARNING: no API key set. "
                  "Set OCR_SPACE_API_KEY or pass api_key=.")

    def recognize(self, bgr_frame):
        ok, buf = cv2.imencode('.jpg', bgr_frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return "", False, "failed to encode frame as JPEG"

        files = {'file': ('scan.jpg', buf.tobytes(), 'image/jpeg')}
        data = {
            'apikey': self.api_key,
            'language': self.language,
            'OCREngine': self.engine,
            'isOverlayRequired': 'true',
            'scale': 'true',
            'detectOrientation': 'false',
        }
        try:
            resp = self._requests.post(self.ENDPOINT, files=files, data=data,
                                        timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
        except self._requests.exceptions.RequestException as e:
            return "", False, f"network/request error: {e}"
        except ValueError as e:
            return "", False, f"bad response (not JSON): {e}"

        if result.get("IsErroredOnProcessing"):
            err = result.get("ErrorMessage", "unknown error")
            if isinstance(err, list):
                err = "; ".join(str(x) for x in err)
            return "", False, f"OCR.space error: {err}"

        parsed = result.get("ParsedResults") or []
        if not parsed:
            return "", False, "no ParsedResults in response"

        pr = parsed[0]
        text = " ".join((pr.get("ParsedText") or "").split())

        edge_touched = False
        frame_w = bgr_frame.shape[1]
        overlay = pr.get("TextOverlay") or {}
        for line in overlay.get("Lines", []) or []:
            for w in line.get("Words", []) or []:
                left = w.get("Left", 0)
                width = w.get("Width", 0)
                if left <= EDGE_MARGIN_PX or (left + width) >= frame_w - EDGE_MARGIN_PX:
                    edge_touched = True
                    break
            if edge_touched:
                break

        return text, edge_touched, ""


def build_backend(name, **kwargs):
    """Factory — the one place that needs to know backend names. Call this
    from the capture script with a single config value/CLI flag; nothing
    else needs to change to switch implementations."""
    name = (name or "").lower()
    if name in ("local", "rapidocr"):
        return LocalRapidOCRBackend(**kwargs)
    if name in ("ocrspace", "ocr.space", "ocr_space"):
        return OCRSpaceBackend(**kwargs)
    raise ValueError(f"Unknown OCR backend: {name!r}")
