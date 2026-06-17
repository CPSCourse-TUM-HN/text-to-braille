import cv2
import pytesseract
from picamera2 import Picamera2

picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": (1920, 1080)}))
picam.start()

while True:
    frame = picam.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Upscale slightly — helps Tesseract with smaller fonts
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Remove noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Binarize
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
    if text.strip():
        print(text.strip())
        print("---")