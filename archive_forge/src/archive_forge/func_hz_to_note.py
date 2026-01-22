from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def hz_to_note(frequencies: _ScalarOrSequence[_FloatLike_co], **kwargs: Any) -> Union[str, np.ndarray]:
    """Convert one or more frequencies (in Hz) to the nearest note names.

    Parameters
    ----------
    frequencies : float or iterable of float
        Input frequencies, specified in Hz
    **kwargs : additional keyword arguments
        Arguments passed through to `midi_to_note`

    Returns
    -------
    notes : str or np.ndarray of str
        ``notes[i]`` is the closest note name to ``frequency[i]``
        (or ``frequency`` if the input is scalar)

    See Also
    --------
    hz_to_midi
    midi_to_note
    note_to_hz

    Examples
    --------
    Get a single note name for a frequency

    >>> librosa.hz_to_note(440.0)
    'A5'

    Get multiple notes with cent deviation

    >>> librosa.hz_to_note([32, 64], cents=True)
    ['C1-38', 'C2-38']

    Get multiple notes, but suppress octave labels

    >>> librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 12)),
    ...                    octave=False)
    ['A', 'A#', 'B', 'C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#', 'A']

    """
    return midi_to_note(hz_to_midi(frequencies), **kwargs)