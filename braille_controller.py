#!/usr/bin/env python3
"""
Braille solenoid controller — drives 6 solenoids representing one Braille
cell.

Three translation modes, selectable at construction time or at runtime via
set_mode():

  - "naive":   the original one-to-one character -> braille cell mapping.
               Uncontracted (Grade 0), no short forms, no external
               dependencies.

  - "grade1":  liblouis, Grade 1 Unified English Braille — every letter
               spelled out, but using real liblouis punctuation/number
               handling rather than the naive table.

  - "grade2":  liblouis, Grade 2 Unified English Braille — contractions
               applied ("ing", "the", "ch", whole-word short forms, etc).
               This is what most modern embossers/refreshable displays use.

There is also a "custom" mode for passing your own liblouis table list
(other languages/locales, math tables, etc) — see `custom_tables` below.

liblouis setup (only needed for "grade1"/"grade2"/"custom" modes):
    pip install louis
    # liblouis's translation tables are a separate data package, not
    # bundled with the python bindings:
    sudo apt-get install liblouis-bin liblouis-data      # Debian/Ubuntu
    # or build from source: https://github.com/liblouis/liblouis

Falls back to a console-printing mock when RPi.GPIO isn't available
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

try:
    import louis
    LIBLOUIS_AVAILABLE = True
except ImportError:
    LIBLOUIS_AVAILABLE = False

# ─── Shared visualization state ───────────────────────────────────────────────
# Read by solenoid_visualizer.py, written by BrailleController whenever a pin
# changes. Lives here (not in the visualizer) so anything — this module's own
# __main__ block, the main reader script, tests — can update it without
# importing the visualizer.
viz_lock  = threading.Lock()
viz_state = {"char": "", "dots": {1: False, 2: False, 3: False,
                                  4: False, 5: False, 6: False}}

# Bit for each dot within a Unicode braille pattern codepoint (U+2800 base).
# Dots 7/8 exist for 8-dot cells but this hardware only drives a 6-dot cell.
_DOT_BITS = {1: 0x01, 2: 0x02, 3: 0x04, 4: 0x08, 5: 0x10, 6: 0x20}
_EXTRA_DOT_BITS = 0x40 | 0x80  # dots 7 and 8


class BrailleController:
    # Built-in modes and the liblouis table list each one uses.
    # "naive" maps to None because it doesn't go through liblouis at all.
    MODES = {
        "naive":  None,
        "grade1": ["en-ueb-g1.ctb"],   # uncontracted Unified English Braille
        "grade2": ["en-ueb-g2.ctb"],   # contracted Unified English Braille
    }

    def __init__(self, mode="naive", custom_tables=None):
        """
        mode:          "naive", "grade1", "grade2", or "custom"
        custom_tables: required if mode="custom" — an explicit list of
                       liblouis table filenames (other languages, math
                       tables, etc).
        """
        # Precise Mapping based on your explicit (Physical Pin, GPIO) pairs
        self.DOT_TO_PIN = {
            1: 17,  # Top Left      (Physical Pin 11)
            2: 27,  # Middle Left   (Physical Pin 13)
            3: 22,  # Bottom Left   (Physical Pin 15)
            4: 14,  # Top Right     (Physical Pin 8)
            5: 15,  # Middle Right  (Physical Pin 10)
            6: 18   # Bottom Right  (Physical Pin 12)
        }

        # Standard English Braille Alphabet (A-Z) mapped to active dot tuples.
        # This is the "naive" mode's entire translation table — kept as-is
        # so that mode still behaves exactly as before.
        self.BRAILLE_ALPHABET = {
            'a': (1,),          'b': (1, 2),       'c': (1, 4),       'd': (1, 4, 5),
            'e': (1, 5),        'f': (1, 2, 4),    'g': (1, 2, 4, 5), 'h': (1, 2, 5),
            'i': (2, 4),        'j': (2, 4, 5),    'k': (1, 3),       'l': (1, 2, 3),
            'm': (1, 3, 4),     'n': (1, 3, 4, 5), 'o': (1, 3, 5),    'p': (1, 2, 3, 4),
            'q': (1, 2, 3, 4, 5),'r': (1, 2, 3, 5), 's': (2, 3, 4),    't': (2, 3, 4, 5),
            'u': (1, 3, 6),     'v': (1, 2, 3, 6), 'w': (2, 4, 5, 6), 'x': (1, 3, 4, 6),
            'y': (1, 3, 4, 5, 6),'z': (1, 3, 5, 6), ' ': ()
        }

        self.custom_tables = custom_tables
        self.set_mode(mode)  # validates + stores self.mode

        self._initialize_gpio()

    # ─── Mode management ──────────────────────────────────────────────────

    def set_mode(self, mode):
        """Switch translation backend. Raises if a liblouis-backed mode is
        requested but the python bindings aren't installed, or if "custom"
        is requested without custom_tables set."""
        if mode == "custom":
            if not self.custom_tables:
                raise ValueError(
                    "mode='custom' requires custom_tables=[...] to be set "
                    "(either in the constructor or by setting "
                    "controller.custom_tables before calling set_mode)."
                )
            if not LIBLOUIS_AVAILABLE:
                raise RuntimeError(
                    "liblouis-backed mode requested but the 'louis' python "
                    "package isn't installed. Run: pip install louis "
                    "(and make sure liblouis-data tables are installed)."
                )
        elif mode in self.MODES:
            if self.MODES[mode] is not None and not LIBLOUIS_AVAILABLE:
                raise RuntimeError(
                    f"mode='{mode}' requires the 'louis' python package, "
                    "which isn't installed. Run: pip install louis "
                    "(and make sure liblouis-data tables are installed)."
                )
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Available: "
                f"{list(self.MODES.keys()) + ['custom']}"
            )
        self.mode = mode

    def _current_tables(self):
        """The liblouis table list for the active mode, or None for naive."""
        if self.mode == "custom":
            return self.custom_tables
        return self.MODES[self.mode]

    # ─── GPIO / low-level actuation ───────────────────────────────────────

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

    def _actuate_dots(self, active_dots, label=""):
        """Shared low-level step: raise/lower the six solenoids to match
        `active_dots` and mirror the change into viz_state. Used by both
        naive single-character display and liblouis cell-by-cell display."""
        if not IS_RPI and label:
            print(f"\n--- Displaying: {label} (Dots: {sorted(active_dots)}) ---")

        for dot, pin in self.DOT_TO_PIN.items():
            GPIO.output(pin, GPIO.HIGH if dot in active_dots else GPIO.LOW)

        with viz_lock:
            viz_state["char"] = label
            for d in viz_state["dots"]:
                viz_state["dots"][d] = d in active_dots

    # ─── Naive mode ────────────────────────────────────────────────────────

    def display_character(self, char):
        """
        Actuates the solenoids to represent a single alphanumeric character
        using the naive (uncontracted) mapping. Returns True if successful,
        False if character is unsupported.
        """
        char = char.lower()
        if char not in self.BRAILLE_ALPHABET:
            print(f"Character '{char}' not supported in this basic mapping.")
            self.clear_cell()
            return False

        active_dots = self.BRAILLE_ALPHABET[char]
        self._actuate_dots(active_dots, label=char.upper())
        return True

    # ─── liblouis modes (grade1 / grade2 / custom) ────────────────────────

    def _unicode_cell_to_dots(self, cell_char):
        """Decode one liblouis output character (a Unicode braille pattern,
        U+2800-U+28FF) into a tuple of active dot numbers (1-6)."""
        bitmask = ord(cell_char) - 0x2800
        if bitmask & _EXTRA_DOT_BITS:
            # 8-dot cell requested (e.g. computer braille / some math tables)
            # but this hardware is 6-dot — dots 7/8 are silently dropped.
            pass
        return tuple(dot for dot, bit in _DOT_BITS.items() if bitmask & bit)

    def translate_to_cells(self, text):
        """Run `text` through liblouis (using the active mode's tables) and
        return a list of dot-tuples, one per braille cell (contractions
        already applied, if the mode supports them)."""
        if not LIBLOUIS_AVAILABLE:
            raise RuntimeError("liblouis python bindings not installed.")
        tables = self._current_tables()
        if tables is None:
            raise RuntimeError("translate_to_cells() requires a liblouis-backed mode")
        braille_unicode = louis.translateString(tables, text)
        return [self._unicode_cell_to_dots(c) for c in braille_unicode]

    # ─── Unified string playback ───────────────────────────────────────────

    def display_string(self, text, delay=1.5):
        """Iterates through `text`, displaying each braille cell with a
        tactile gap. Routes to the naive or liblouis backend depending on
        self.mode."""
        if self.mode == "naive":
            self._display_string_naive(text, delay)
        else:
            self._display_string_liblouis(text, delay)

    def _display_string_naive(self, text, delay):
        for char in text:
            success = self.display_character(char)
            if success:
                time.sleep(delay)
                self.clear_cell()
                time.sleep(0.2)  # Short tactile pause between characters

    def _display_string_liblouis(self, text, delay):
        # NOTE: translation happens on the whole string up front, not
        # character-by-character — contractions like "the" or "ing" only
        # exist once liblouis sees the surrounding letters.
        cells = self.translate_to_cells(text)
        for dots in cells:
            self._actuate_dots(dots, label=f"dots {sorted(dots)}")
            time.sleep(delay)
            self.clear_cell()
            time.sleep(0.2)

    def cleanup(self):
        """Safely clears pins and releases GPIO resources."""
        self.clear_cell()
        GPIO.cleanup()


# --- Execution Example ---
if __name__ == "__main__":
    test_word = "braille"

    print("=== naive mode (uncontracted, letter-by-letter) ===")
    controller = BrailleController(mode="naive")
    try:
        print(f"Starting tactile playback for: '{test_word}'")
        controller.display_string(test_word, delay=1.2)
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        controller.cleanup()

    if LIBLOUIS_AVAILABLE:
        for mode in ("grade1", "grade2"):
            print(f"\n=== {mode} mode (liblouis) ===")
            controller = BrailleController(mode=mode)
            try:
                print(f"Starting tactile playback for: '{test_word}'")
                controller.display_string(test_word, delay=1.2)
            except KeyboardInterrupt:
                print("\nPlayback interrupted by user.")
            finally:
                controller.cleanup()
    else:
        print("\n(liblouis not installed — skipping grade1/grade2 demos. "
              "Run `pip install louis` to try them.)")