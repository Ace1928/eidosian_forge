import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def qpm_to_bpm(quarter_note_tempo, numerator, denominator):
    """Converts from quarter notes per minute to beats per minute.

    Parameters
    ----------
    quarter_note_tempo : float
        Quarter note tempo.
    numerator : int
        Numerator of time signature.
    denominator : int
        Denominator of time signature.

    Returns
    -------
    bpm : float
        Tempo in beats per minute.
    """
    if not (isinstance(quarter_note_tempo, (int, float)) and quarter_note_tempo > 0):
        raise ValueError('Quarter notes per minute must be an int or float greater than 0, but {} was supplied'.format(quarter_note_tempo))
    if not (isinstance(numerator, int) and numerator > 0):
        raise ValueError('Time signature numerator must be an int greater than 0, but {} was supplied.'.format(numerator))
    if not (isinstance(denominator, int) and denominator > 0):
        raise ValueError('Time signature denominator must be an int greater than 0, but {} was supplied.'.format(denominator))
    if denominator in [1, 2, 4, 8, 16, 32]:
        if numerator == 3:
            return quarter_note_tempo * denominator / 4.0
        elif numerator % 3 == 0:
            return quarter_note_tempo / 3.0 * denominator / 4.0
        else:
            return quarter_note_tempo * denominator / 4.0
    else:
        return quarter_note_tempo