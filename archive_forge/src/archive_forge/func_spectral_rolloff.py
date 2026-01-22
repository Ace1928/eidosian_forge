import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from .. import util
from .. import filters
from ..util.exceptions import ParameterError
from ..core.convert import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import power_to_db, _spectrogram
from ..core.constantq import cqt, hybrid_cqt, vqt
from ..core.pitch import estimate_tuning
from typing import Any, Optional, Union, Collection
from numpy.typing import DTypeLike
from .._typing import _FloatLike_co, _WindowSpec, _PadMode, _PadModeSTFT
def spectral_rolloff(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', freq: Optional[np.ndarray]=None, roll_percent: float=0.85) -> np.ndarray:
    """Compute roll-off frequency.

    The roll-off frequency is defined for each frame as the center frequency
    for a spectrogram bin such that at least roll_percent (0.85 by default)
    of the energy of the spectrum in this frame is contained in this bin and
    the bins below. This can be used to, e.g., approximate the maximum (or
    minimum) frequency by setting roll_percent to a value close to 1 (or 0).

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    freq : None or np.ndarray [shape=(d,) or shape=(..., d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        .. note:: ``freq`` is assumed to be sorted in increasing order
    roll_percent : float [0 < roll_percent < 1]
        Roll-off percentage.

    Returns
    -------
    rolloff : np.ndarray [shape=(..., 1, t)]
        roll-off frequency for each frame

    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> # Approximate maximum frequencies with roll_percent=0.85 (default)
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr)
    array([[2583.984, 3036.182, ..., 9173.145, 9248.511]])
    >>> # Approximate maximum frequencies with roll_percent=0.99
    >>> rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
    >>> rolloff
    array([[ 7192.09 ,  6739.893, ..., 10960.4  , 10992.7  ]])
    >>> # Approximate minimum frequencies with roll_percent=0.01
    >>> rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)
    >>> rolloff_min
    array([[516.797, 538.33 , ..., 764.429, 764.429]])

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> librosa.feature.spectral_rolloff(S=S, sr=sr)
    array([[2583.984, 3036.182, ..., 9173.145, 9248.511]])

    >>> # With a higher roll percentage:
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    array([[ 3919.043,  3994.409, ..., 10443.604, 10594.336]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax)
    >>> ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
    >>> ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
    ...         label='Roll-off frequency (0.01)')
    >>> ax.legend(loc='lower right')
    >>> ax.set(title='log Power spectrogram')
    """
    if not 0.0 < roll_percent < 1.0:
        raise ParameterError('roll_percent must lie in the range (0, 1)')
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    if not np.isrealobj(S):
        raise ParameterError('Spectral rolloff is only defined with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral rolloff is only defined with non-negative energies')
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)
    if freq.ndim == 1:
        freq = util.expand_to(freq, ndim=S.ndim, axes=-2)
    total_energy = np.cumsum(S, axis=-2)
    threshold = roll_percent * total_energy[..., -1, :]
    threshold = np.expand_dims(threshold, axis=-2)
    ind = np.where(total_energy < threshold, np.nan, 1)
    rolloff: np.ndarray = np.nanmin(ind * freq, axis=-2, keepdims=True)
    return rolloff