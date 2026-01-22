import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def program_to_instrument_class(program_number):
    """Converts a MIDI program number to the corresponding General MIDI
    instrument class.

    Parameters
    ----------
    program_number : int
        MIDI program number, between 0 and 127.

    Returns
    -------
    instrument_class : str
        Name of the instrument class corresponding to this program number.

    Notes
    -----
        See http://www.midi.org/techspecs/gm1sound.php

    """
    if program_number < 0 or program_number > 127:
        raise ValueError('Invalid program number {}, should be between 0 and 127'.format(program_number))
    return INSTRUMENT_CLASSES[int(program_number) // 8]