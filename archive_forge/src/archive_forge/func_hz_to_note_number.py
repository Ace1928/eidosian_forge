import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def hz_to_note_number(frequency):
    """Convert a frequency in Hz to a (fractional) note number.

    Parameters
    ----------
    frequency : float
        Frequency of the note in Hz.

    Returns
    -------
    note_number : float
        MIDI note number, can be fractional.

    """
    return 12 * (np.log2(frequency) - np.log2(440.0)) + 69