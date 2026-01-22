from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def hz_to_fjs(frequencies: _ScalarOrSequence[_FloatLike_co], *, fmin: Optional[float]=None, unison: Optional[str]=None, unicode: bool=False) -> Union[str, np.ndarray]:
    """Convert one or more frequencies (in Hz) from a just intonation
    scale to notes in FJS notation.

    Parameters
    ----------
    frequencies : float or iterable of float
        Input frequencies, specified in Hz
    fmin : float (optional)
        The minimum frequency, corresponding to a unison note.
        If not provided, it will be inferred as `min(frequencies)`
    unison : str (optional)
        The name of the unison note.
        If not provided, it will be inferred as the scientific pitch
        notation name of `fmin`, that is, `hz_to_note(fmin)`
    unicode : bool
        If `True`, then unicode symbols are used for accidentals.
        If `False`, then low-order ASCII symbols are used for accidentals.

    Returns
    -------
    notes : str or np.ndarray(dtype=str)
        ``notes[i]`` is the closest note name to ``frequency[i]``
        (or ``frequency`` if the input is scalar)

    See Also
    --------
    hz_to_note
    interval_to_fjs

    Examples
    --------
    Get a single note name for a frequency, relative to A=55 Hz

    >>> librosa.hz_to_fjs(66, fmin=55, unicode=True)
    'C₅'

    Get notation for a 5-limit frequency set starting at A=55

    >>> freqs = librosa.interval_frequencies(24, intervals="ji5", fmin=55)
    >>> freqs
    array([ 55.   ,  58.667,  61.875,  66.   ,  68.75 ,  73.333,  77.344,
        82.5  ,  88.   ,  91.667,  99.   , 103.125, 110.   , 117.333,
       123.75 , 132.   , 137.5  , 146.667, 154.687, 165.   , 176.   ,
       183.333, 198.   , 206.25 ])
    >>> librosa.hz_to_fjs(freqs, unicode=True)
    array(['A', 'B♭₅', 'B', 'C₅', 'C♯⁵', 'D', 'D♯⁵', 'E', 'F₅', 'F♯⁵', 'G₅',
       'G♯⁵', 'A', 'B♭₅', 'B', 'C₅', 'C♯⁵', 'D', 'D♯⁵', 'E', 'F₅', 'F♯⁵',
       'G₅', 'G♯⁵'], dtype='<U3')

    """
    if fmin is None:
        fmin = np.min(frequencies)
    if unison is None:
        unison = hz_to_note(fmin, octave=False, unicode=False)
    if np.isscalar(frequencies):
        intervals = frequencies / fmin
    else:
        intervals = np.asarray(frequencies) / fmin
    return notation.interval_to_fjs(intervals, unison=unison, unicode=unicode)