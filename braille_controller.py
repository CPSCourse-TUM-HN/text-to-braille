#!/usr/bin/env python3
"""
Braille solenoid controller — drives 6 solenoids representing one Braille
cell. Falls back to a console-printing mock when RPi.GPIO isn't available
(e.g. developing off the Pi). Every pin change also updates `viz_state`
below, which solenoid_visualizer.py reads to render a live view of what the
cell would be doing without needing the physical hardware wired up.
"""

import time
import threading

try:
    import RPi.GPIO as GPIO
    IS_RPI = True
except ImportError:
    IS_RPI = False
    # Mock class mimicking RPi.GPIO behavior for local testing
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        def setmode(self, mode): print(f"[MOCK] GPIO Mode set to {mode}")
        def setup(self, pin, mode): print(f"[MOCK] Pin {pin} configured as {mode}")
        def output(self, pin, state): print(f"[MOCK] Pin {pin} -> {'HIGH' if state else 'LOW'}")
        def cleanup(self): print("[MOCK] GPIO Cleanup complete")
    GPIO = MockGPIO()

# ─── Shared visualization state ───────────────────────────────────────────────
# Read by solenoid_visualizer.py, written by BrailleController whenever a pin
# changes. Lives here (not in the visualizer) so anything — this module's own
# __main__ block, the main reader script, tests — can update it without
# importing the visualizer.
viz_lock  = threading.Lock()
viz_state = {"char": "", "dots": {1: False, 2: False, 3: False,
                                  4: False, 5: False, 6: False}}


class BrailleController:
    def __init__(self):
        # Precise Mapping based on your explicit (Physical Pin, GPIO) pairs
        self.DOT_TO_PIN = {
            1: 17,  # Top Left      (Physical Pin 11)
            2: 27,  # Middle Left   (Physical Pin 13)
            3: 22,  # Bottom Left   (Physical Pin 15)
            4: 14,  # Top Right     (Physical Pin 8)
            5: 15,  # Middle Right  (Physical Pin 10)
            6: 18   # Bottom Right  (Physical Pin 12)
        }

        # Standard English Braille Alphabet (A-Z) mapped to active dot tuples
        self.BRAILLE_ALPHABET = {
            'a': (1,),          'b': (1, 2),       'c': (1, 4),       'd': (1, 4, 5),
            'e': (1, 5),        'f': (1, 2, 4),    'g': (1, 2, 4, 5), 'h': (1, 2, 5),
            'i': (2, 4),        'j': (2, 4, 5),    'k': (1, 3),       'l': (1, 2, 3),
            'm': (1, 3, 4),     'n': (1, 3, 4, 5), 'o': (1, 3, 5),    'p': (1, 2, 3, 4),
            'q': (1, 2, 3, 4, 5),'r': (1, 2, 3, 5), 's': (2, 3, 4),    't': (2, 3, 4, 5),
            'u': (1, 3, 6),     'v': (1, 2, 3, 6), 'w': (2, 4, 5, 6), 'x': (1, 3, 4, 6),
            'y': (1, 3, 4, 5, 6),'z': (1, 3, 5, 6), ' ': ()
        }

        self._initialize_gpio()

    def _initialize_gpio(self):
        """Initializes the GPIO layout and configures pins as outputs."""
        GPIO.setmode(GPIO.BCM)
        for pin in self.DOT_TO_PIN.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)  # Ensure starting state is off

    def clear_cell(self):
        """Turns off all solenoids."""
        for pin in self.DOT_TO_PIN.values():
            GPIO.output(pin, GPIO.LOW)
        with viz_lock:
            viz_state["char"] = ""
            for d in viz_state["dots"]:
                viz_state["dots"][d] = False

    def display_character(self, char):
        """
        Actuates the solenoids to represent a single alphanumeric character.
        Returns True if successful, False if character is unsupported.
        """
        char = char.lower()
        if char not in self.BRAILLE_ALPHABET:
            print(f"Character '{char}' not supported in this basic mapping.")
            self.clear_cell()
            return False

        active_dots = self.BRAILLE_ALPHABET[char]

        if not IS_RPI:
            print(f"\n--- Displaying Character: '{char.upper()}' (Dots: {active_dots}) ---")

        # Iterate through all 6 dots and actuate accordingly
        for dot, pin in self.DOT_TO_PIN.items():
            if dot in active_dots:
                GPIO.output(pin, GPIO.HIGH)
            else:
                GPIO.output(pin, GPIO.LOW)

        with viz_lock:
            viz_state["char"] = char
            for d in viz_state["dots"]:
                viz_state["dots"][d] = d in active_dots

        return True

    def display_string(self, text, delay=1.5):
        """Iterates through a string, displaying each letter with a tactile gap."""
        for char in text:
            success = self.display_character(char)
            if success:
                time.sleep(delay)
                self.clear_cell()
                time.sleep(0.2)  # Short tactile pause between characters

    def cleanup(self):
        """Safely clears pins and releases GPIO resources."""
        self.clear_cell()
        GPIO.cleanup()


# --- Execution Example ---
if __name__ == "__main__":
    controller = BrailleController()
    try:
        test_word = "braille"
        print(f"Starting tactile playback for: '{test_word}'")
        controller.display_string(test_word, delay=1.2)

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        controller.cleanup()
