import re
import numpy as np
from numba import jit
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike
@cache(level=10)
def fifths_to_note(*, unison: str, fifths: int, unicode: bool=True) -> str:
    """Calculate the note name for a given number of perfect fifths
    from a specified unison.

    This function is primarily intended as a utility routine for
    Functional Just System (FJS) notation conversions.

    This function does not assume the "circle of fifths" or equal temperament,
    so 12 fifths will not generally produce a note of the same pitch class
    due to the accumulation of accidentals.

    Parameters
    ----------
    unison : str
        The name of the starting (unison) note, e.g., 'C' or 'Bb'.
        Unicode accidentals are supported.

    fifths : integer
        The number of perfect fifths to deviate from unison.

    unicode : bool
        If ``True`` (default), use Unicode symbols (♯𝄪♭𝄫)for accidentals.

        If ``False``, accidentals will be encoded as low-order ASCII representations::

            ♯ -> #, 𝄪 -> ##, ♭ -> b, 𝄫 -> bb

    Returns
    -------
    note : str
        The name of the requested note

    Examples
    --------
    >>> librosa.fifths_to_note(unison='C', fifths=6)
    'F♯'

    >>> librosa.fifths_to_note(unison='G', fifths=-3)
    'B♭'

    >>> librosa.fifths_to_note(unison='Eb', fifths=11, unicode=False)
    'G#'
    """
    COFMAP = 'FCGDAEB'
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1, '♯': 1, '𝄪': 2, '♭': -1, '𝄫': -2, '♮': 0}
    if unicode:
        acc_map_inv = {1: '♯', 2: '𝄪', -1: '♭', -2: '𝄫', 0: ''}
    else:
        acc_map_inv = {1: '#', 2: '##', -1: 'b', -2: 'bb', 0: ''}
    match = NOTE_RE.match(unison)
    if not match:
        raise ParameterError(f'Improper note format: {unison:s}')
    pitch = match.group('note').upper()
    offset = np.sum([acc_map[o] for o in match.group('accidental')])
    circle_idx = COFMAP.index(pitch)
    raw_output = COFMAP[(circle_idx + fifths) % 7]
    acc_index = offset + (circle_idx + fifths) // 7
    acc_str = acc_map_inv[np.sign(acc_index) * 2] * int(abs(acc_index) // 2) + acc_map_inv[np.sign(acc_index)] * int(abs(acc_index) % 2)
    return raw_output + acc_str