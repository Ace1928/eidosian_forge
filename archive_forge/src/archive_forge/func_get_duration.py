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
def get_duration(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, center: bool=True, path: Optional[Union[str, os.PathLike[Any]]]=None, filename: Optional[Union[str, os.PathLike[Any], Deprecated]]=Deprecated()) -> float:
    """Compute the duration (in seconds) of an audio time series,
    feature matrix, or filename.

    Examples
    --------
    >>> # Load an example audio file
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.get_duration(y=y, sr=sr)
    5.333378684807256

    >>> # Or directly from an audio file
    >>> librosa.get_duration(filename=librosa.ex('trumpet'))
    5.333378684807256

    >>> # Or compute duration from an STFT matrix
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = librosa.stft(y)
    >>> librosa.get_duration(S=S, sr=sr)
    5.317369614512471

    >>> # Or a non-centered STFT matrix
    >>> S_left = librosa.stft(y, center=False)
    >>> librosa.get_duration(S=S_left, sr=sr)
    5.224489795918367

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        audio sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        STFT matrix, or any STFT-derived matrix (e.g., chromagram
        or mel spectrogram).
        Durations calculated from spectrogram inputs are only accurate
        up to the frame resolution. If high precision is required,
        it is better to use the audio time series directly.

    n_fft : int > 0 [scalar]
        FFT window size for ``S``

    hop_length : int > 0 [ scalar]
        number of audio samples between columns of ``S``

    center : boolean
        - If ``True``, ``S[:, t]`` is centered at ``y[t * hop_length]``
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``

    path : str, path, or file-like
        If provided, all other parameters are ignored, and the
        duration is calculated directly from the audio file.
        Note that this avoids loading the contents into memory,
        and is therefore useful for querying the duration of
        long files.

        As in ``load``, this can also be an integer or open file-handle
        that can be processed by ``soundfile``.

    filename : Deprecated
        Equivalent to ``path``

        .. warning:: This parameter has been renamed to ``path`` in 0.10.
            Support for ``filename=`` will be removed in 1.0.

    Returns
    -------
    d : float >= 0
        Duration (in seconds) of the input time series or spectrogram.

    Raises
    ------
    ParameterError
        if none of ``y``, ``S``, or ``path`` are provided.

    Notes
    -----
    `get_duration` can be applied to a file (``path``), a spectrogram (``S``),
    or audio buffer (``y, sr``).  Only one of these three options should be
    provided.  If you do provide multiple options (e.g., ``path`` and ``S``),
    then ``path`` takes precedence over ``S``, and ``S`` takes precedence over
    ``(y, sr)``.
    """
    path = rename_kw(old_name='filename', old_value=filename, new_name='path', new_value=path, version_deprecated='0.10.0', version_removed='1.0')
    if path is not None:
        try:
            return sf.info(path).duration
        except sf.SoundFileRuntimeError:
            warnings.warn('PySoundFile failed. Trying audioread instead.\n\tAudioread support is deprecated in librosa 0.10.0 and will be removed in version 1.0.', stacklevel=2, category=FutureWarning)
            with audioread.audio_open(path) as fdesc:
                return fdesc.duration
    if y is None:
        if S is None:
            raise ParameterError('At least one of (y, sr), S, or path must be provided')
        n_frames = S.shape[-1]
        n_samples = n_fft + hop_length * (n_frames - 1)
        if center:
            n_samples = n_samples - 2 * int(n_fft // 2)
    else:
        n_samples = y.shape[-1]
    return float(n_samples) / sr