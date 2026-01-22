from __future__ import annotations
import os
import pathlib
import warnings
import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import soxr
import lazy_loader as lazy
from numba import jit, stencil, guvectorize
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecated
from ..util.deprecation import Deprecated, rename_kw
from .._typing import _FloatLike_co, _IntLike_co, _SequenceLike
from typing import Any, BinaryIO, Callable, Generator, Optional, Tuple, Union, List
from numpy.typing import DTypeLike, ArrayLike
def tone(frequency: _FloatLike_co, *, sr: float=22050, length: Optional[int]=None, duration: Optional[float]=None, phi: Optional[float]=None) -> np.ndarray:
    """Construct a pure tone (cosine) signal at a given frequency.

    Parameters
    ----------
    frequency : float > 0
        frequency
    sr : number > 0
        desired sampling rate of the output signal
    length : int > 0
        desired number of samples in the output signal.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    duration : float > 0
        desired duration in seconds.
        When both ``duration`` and ``length`` are defined,
        ``length`` takes priority.
    phi : float or None
        phase offset, in radians. If unspecified, defaults to ``-np.pi * 0.5``.

    Returns
    -------
    tone_signal : np.ndarray [shape=(length,), dtype=float64]
        Synthesized pure sine tone signal

    Raises
    ------
    ParameterError
        - If ``frequency`` is not provided.
        - If neither ``length`` nor ``duration`` are provided.

    Examples
    --------
    Generate a pure sine tone A4

    >>> tone = librosa.tone(440, duration=1)

    Or generate the same signal using `length`

    >>> tone = librosa.tone(440, sr=22050, length=22050)

    Display spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S = librosa.feature.melspectrogram(y=tone)
    >>> librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    ...                          x_axis='time', y_axis='mel', ax=ax)
    """
    if frequency is None:
        raise ParameterError('"frequency" must be provided')
    if length is None:
        if duration is None:
            raise ParameterError('either "length" or "duration" must be provided')
        length = int(duration * sr)
    if phi is None:
        phi = -np.pi * 0.5
    y: np.ndarray = np.cos(2 * np.pi * frequency * np.arange(length) / sr + phi)
    return y