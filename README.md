# text-to-braille

A camera-based reading aid that captures a line of printed text, recognizes it with OCR, and outputs it on a physical 6-dot Braille cell driven by solenoids. Built for the Raspberry Pi.

Point the device at text, press a key, and the line is read aloud in Braille — dot by dot, in the order it was captured.

## How it works

```
Camera Capture → Preprocessing → OCR Backend → Braille Mapping → Tactile Output
```

1. **Camera Capture** — a Raspberry Pi camera (via Picamera2) continuously streams a low-res preview and grabs a full-resolution frame on trigger.
2. **Preprocessing** — the captured frame is cleaned up (deskew, lighting correction, upscaling if needed) before recognition.
3. **OCR Backend** — recognizes text using either an on-device engine (RapidOCR) or a cloud API (OCR.space), selectable at runtime.
4. **Braille Mapping** — recognized text is translated into Braille cells, either via a built-in one-to-one table or via `liblouis` for Grade 1/Grade 2 Unified English Braille.
5. **Tactile Output** — each cell's dot pattern is played back on the physical 6-solenoid Braille cell, one character at a time.

Two diagnostic web views run alongside the pipeline:
- **Live camera feed** — `http://<pi-ip>:8000`
- **Virtual Braille cell** (mirrors the physical solenoids, useful without hardware attached) — `http://<pi-ip>:8001`

## Hardware

- Raspberry Pi (with camera support via Picamera2/libcamera)
- Pi Camera module
- 6 solenoids wired to GPIO, one per Braille dot

| Dot | Position     | BCM GPIO |
|-----|--------------|----------|
| 1   | Top left     | 17       |
| 2   | Middle left  | 27       |
| 3   | Bottom left  | 22       |
| 4   | Top right    | 14       |
| 5   | Middle right | 15       |
| 6   | Bottom right | 18       |

No hardware attached? `BrailleController` transparently falls back to a mock GPIO layer and mirrors dot states to the virtual Braille cell web view, so the full pipeline runs on a laptop for development.

## Installation

```bash
git clone https://github.com/aoprea42/text-to-braille.git
cd text-to-braille
pip install -r requirements.txt
```

You'll also need the system-level `liblouis` packages listed in `apt-packages.txt`, since the Python `louis` bindings link against them:

```bash
sudo apt install liblouis-bin liblouis-data
```

On the Raspberry Pi you'll also need `picamera2` and `libcamera` set up (these ship with Raspberry Pi OS by default). RPi.GPIO is required to drive the physical solenoids; it's optional for development off-device.

### OCR.space API key (optional)

Only needed if you plan to use the cloud OCR backend. Get a free key at [ocr.space](https://ocr.space/ocrapi) and set it as an environment variable:

```bash
export OCR_SPACE_API_KEY=your_key_here
```

## Usage

```bash
python reader_main.py
```

Then point the camera at a line of text and press **space** or **enter** to capture and read it. Press **q** to quit.

### Options

| Flag            | Default | Description                                   |
|------------------|---------|------------------------------------------------|
| `--camera`       | `0`     | Camera index                                   |
| `--camera-port`  | `8000`  | Port for the live camera view                  |
| `--viz-port`     | `8001`  | Port for the virtual Braille cell view         |
| `--backend`      | `local` | OCR backend: `local` (RapidOCR) or `ocrspace` (OCR.space API) |

```bash
# Use the cloud OCR backend instead of the local model
python reader_main.py --backend ocrspace
```

Braille translation mode (`naive`, `grade1`, or `grade2`) is set via the `BRAILLE_MODE` constant in `reader_main.py`.

## Project structure

```
reader_main.py          # capture loop, keyboard trigger, camera/viz web servers, main entry point
ocr_backend.py           # OCRBackend interface + LocalRapidOCRBackend / OCRSpaceBackend implementations
braille_controller.py    # Braille translation (naive / liblouis) and solenoid actuation
solenoid_visualizer.py   # web-based virtual Braille cell for development without hardware
requirements.txt         # Python dependencies
```

## Notes

- Capture is user-triggered rather than continuous, with a short cooldown to avoid duplicate reads from a single keypress.
- Braille playback runs on its own worker thread so a multi-second tactile readout never blocks the camera preview or the next capture.
- Grade 2 (contracted) Braille requires translating the whole recognized line at once — contractions like "the" or "-ing" only resolve correctly when `liblouis` has the surrounding context, so this mode can't operate character-by-character.
