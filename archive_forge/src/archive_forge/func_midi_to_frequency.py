from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def midi_to_frequency(midi_note):
    """Converts a midi note to a frequency.

    ::Examples::

    >>> midi_to_frequency(21)
    27.5
    >>> midi_to_frequency(26)
    36.7
    >>> midi_to_frequency(108)
    4186.0
    """
    return round(440.0 * 2 ** ((midi_note - 69) * (1.0 / 12.0)), 1)