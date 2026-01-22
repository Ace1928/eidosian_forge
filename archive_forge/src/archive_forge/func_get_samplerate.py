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
def get_samplerate(path: Union[str, int, sf.SoundFile, BinaryIO]) -> float:
    """Get the sampling rate for a given file.

    Parameters
    ----------
    path : string, int, soundfile.SoundFile, or file-like
        The path to the file to be loaded
        As in ``load``, this can also be an integer or open file-handle
        that can be processed by `soundfile`.
        An existing `soundfile.SoundFile` object can also be supplied.

    Returns
    -------
    sr : number > 0
        The sampling rate of the given audio file

    Examples
    --------
    Get the sampling rate for the included audio file

    >>> path = librosa.ex('trumpet')
    >>> librosa.get_samplerate(path)
    22050
    """
    try:
        if isinstance(path, sf.SoundFile):
            return path.samplerate
        return sf.info(path).samplerate
    except sf.SoundFileRuntimeError:
        warnings.warn('PySoundFile failed. Trying audioread instead.\n\tAudioread support is deprecated in librosa 0.10.0 and will be removed in version 1.0.', stacklevel=2, category=FutureWarning)
        with audioread.audio_open(path) as fdesc:
            return fdesc.samplerate