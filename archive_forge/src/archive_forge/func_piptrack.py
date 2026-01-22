import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
@cache(level=30)
def piptrack(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: Optional[int]=2048, hop_length: Optional[int]=None, fmin: float=150.0, fmax: float=4000.0, threshold: float=0.1, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', ref: Optional[Union[float, Callable]]=None) -> Tuple[np.ndarray, np.ndarray]:
    """Pitch tracking on thresholded parabolically-interpolated STFT.

    This implementation uses the parabolic interpolation method described by [#]_.

    .. [#] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio signal. Multi-channel is supported..

    sr : number > 0 [scalar]
        audio sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        magnitude or power spectrogram

    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if ``y`` is provided.

    hop_length : int > 0 [scalar] or None
        number of samples to hop

    threshold : float in `(0, 1)`
        A bin in spectrum ``S`` is considered a pitch when it is greater than
        ``threshold * ref(S)``.

        By default, ``ref(S)`` is taken to be ``max(S, axis=0)`` (the maximum value in
        each column).

    fmin : float > 0 [scalar]
        lower frequency cutoff.

    fmax : float > 0 [scalar]
        upper frequency cutoff.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero-padding.

        See also: `np.pad`.

    ref : scalar or callable [default=np.max]
        If scalar, the reference value against which ``S`` is compared for determining
        pitches.

        If callable, the reference value is computed as ``ref(S, axis=0)``.

    Returns
    -------
    pitches, magnitudes : np.ndarray [shape=(..., d, t)]
        Where ``d`` is the subset of FFT bins within ``fmin`` and ``fmax``.

        ``pitches[..., f, t]`` contains instantaneous frequency at bin
        ``f``, time ``t``

        ``magnitudes[..., f, t]`` contains the corresponding magnitudes.

        Both ``pitches`` and ``magnitudes`` take value 0 at bins
        of non-maximal magnitude.

    Notes
    -----
    This function caches at level 30.

    One of ``S`` or ``y`` must be provided.
    If ``S`` is not given, it is computed from ``y`` using
    the default parameters of `librosa.stft`.

    Examples
    --------
    Computing pitches from a waveform input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    Or from a spectrogram input

    >>> S = np.abs(librosa.stft(y))
    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr)

    Or with an alternate reference value for pitch detection, where
    values above the mean spectral energy in each frame are counted as pitches

    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr, threshold=1,
    ...                                        ref=np.mean)
    """
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    S = np.abs(S)
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)
    fft_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
    avg = np.gradient(S, axis=-2)
    shift = _parabolic_interpolation(S, axis=-2)
    dskew = 0.5 * avg * shift
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)
    freq_mask = (fmin <= fft_freqs) & (fft_freqs < fmax)
    freq_mask = util.expand_to(freq_mask, ndim=S.ndim, axes=-2)
    if ref is None:
        ref = np.max
    if callable(ref):
        ref_value = threshold * ref(S, axis=-2)
        ref_value = np.expand_dims(ref_value, -2)
    else:
        ref_value = np.abs(ref)
    idx = np.nonzero(freq_mask & util.localmax(S * (S > ref_value), axis=-2))
    pitches[idx] = (idx[-2] + shift[idx]) * float(sr) / n_fft
    mags[idx] = S[idx] + dskew[idx]
    return (pitches, mags)