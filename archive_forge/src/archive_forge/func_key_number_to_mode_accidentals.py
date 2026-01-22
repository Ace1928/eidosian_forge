import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def key_number_to_mode_accidentals(key_number):
    """Converts a key number to number of accidentals and mode.

    Parameters
    ----------
    key_number : int
        Key number as used in ``pretty_midi``.

    Returns
    -------
    mode : int
        0 for major, 1 for minor.
    num_accidentals : int
        Number of accidentals.
        Positive is for sharps and negative is for flats.
    """
    if not (isinstance(key_number, int) and key_number >= 0 and (key_number < 24)):
        raise ValueError('Key number {} is not a must be an int between 0 and 24'.format(key_number))
    pc_to_num_accidentals_major = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: -1, 6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}
    mode = key_number // 12
    if mode == 0:
        num_accidentals = pc_to_num_accidentals_major[key_number]
        return (mode, num_accidentals)
    elif mode == 1:
        key_number = (key_number + 3) % 12
        num_accidentals = pc_to_num_accidentals_major[key_number]
        return (mode, num_accidentals)
    else:
        return None