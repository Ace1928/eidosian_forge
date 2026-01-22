from __future__ import annotations
import re
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._typing import (
def tempo_frequencies(n_bins: int, *, hop_length: int=512, sr: float=22050) -> np.ndarray:
    """Compute the frequencies (in beats per minute) corresponding
    to an onset auto-correlation or tempogram matrix.

    Parameters
    ----------
    n_bins : int > 0
        The number of lag bins
    hop_length : int > 0
        The number of samples between each bin
    sr : number > 0
        The audio sampling rate

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_bins,)]
        vector of bin frequencies measured in BPM.

        .. note:: ``bin_frequencies[0] = +np.inf`` corresponds to 0-lag

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >>> librosa.tempo_frequencies(384)
    array([      inf,  2583.984,  1291.992, ...,     6.782,
               6.764,     6.747])
    """
    bin_frequencies = np.zeros(int(n_bins), dtype=np.float64)
    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))
    return bin_frequencies