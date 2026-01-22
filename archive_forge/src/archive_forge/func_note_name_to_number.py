import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def note_name_to_number(note_name):
    """Converts a note name in the format
    ``'(note)(accidental)(octave number)'`` (e.g. ``'C#4'``) to MIDI note
    number.

    ``'(note)'`` is required, and is case-insensitive.

    ``'(accidental)'`` should be ``''`` for natural, ``'#'`` for sharp and
    ``'!'`` or ``'b'`` for flat.

    If ``'(octave)'`` is ``''``, octave 0 is assumed.

    Parameters
    ----------
    note_name : str
        A note name, as described above.

    Returns
    -------
    note_number : int
        MIDI note number corresponding to the provided note name.

    Notes
    -----
        Thanks to Brian McFee.

    """
    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1}
    try:
        match = re.match('^(?P<n>[A-Ga-g])(?P<off>[#b!]?)(?P<oct>[+-]?\\d+)$', note_name)
        pitch = match.group('n').upper()
        offset = acc_map[match.group('off')]
        octave = int(match.group('oct'))
    except:
        raise ValueError('Improper note format: {}'.format(note_name))
    return 12 * (octave + 1) + pitch_map[pitch] + offset