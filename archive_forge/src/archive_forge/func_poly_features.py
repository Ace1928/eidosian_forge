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
def poly_features(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', order: int=1, freq: Optional[np.ndarray]=None) -> np.ndarray:
    """Get coefficients of fitting an nth-order polynomial to the columns
    of a spectrogram.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
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
          `t` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    order : int > 0
        order of the polynomial to fit
    freq : None or np.ndarray [shape=(d,) or shape=(..., d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`

    Returns
    -------
    coefficients : np.ndarray [shape=(..., order+1, t)]
        polynomial coefficients for each frame.

        ``coefficients[..., 0, :]`` corresponds to the highest degree (``order``),

        ``coefficients[..., 1, :]`` corresponds to the next highest degree (``order-1``),

        down to the constant term ``coefficients[..., order, :]``.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))

    Fit a degree-0 polynomial (constant) to each frame

    >>> p0 = librosa.feature.poly_features(S=S, order=0)

    Fit a linear polynomial to each frame

    >>> p1 = librosa.feature.poly_features(S=S, order=1)

    Fit a quadratic to each frame

    >>> p2 = librosa.feature.poly_features(S=S, order=2)

    Plot the results for comparison

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
    >>> times = librosa.times_like(p0)
    >>> ax[0].plot(times, p0[0], label='order=0', alpha=0.8)
    >>> ax[0].plot(times, p1[1], label='order=1', alpha=0.8)
    >>> ax[0].plot(times, p2[2], label='order=2', alpha=0.8)
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> ax[0].set(ylabel='Constant term ')
    >>> ax[1].plot(times, p1[0], label='order=1', alpha=0.8)
    >>> ax[1].plot(times, p2[1], label='order=2', alpha=0.8)
    >>> ax[1].set(ylabel='Linear term')
    >>> ax[1].label_outer()
    >>> ax[1].legend()
    >>> ax[2].plot(times, p2[0], label='order=2', alpha=0.8)
    >>> ax[2].set(ylabel='Quadratic term')
    >>> ax[2].legend()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[3])
    """
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)
    coefficients: np.ndarray
    if freq.ndim == 1:
        fitter = np.vectorize(lambda y: np.polyfit(freq, y, order), signature='(f,t)->(d,t)')
        coefficients = fitter(S)
    else:
        fitter = np.vectorize(lambda x, y: np.polyfit(x, y, order), signature='(f),(f)->(d)')
        coefficients = fitter(freq.swapaxes(-2, -1), S.swapaxes(-2, -1)).swapaxes(-2, -1)
    return coefficients