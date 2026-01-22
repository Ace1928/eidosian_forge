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
def chroma_stft(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, norm: Optional[float]=np.inf, n_fft: int=2048, hop_length: int=512, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', tuning: Optional[float]=None, n_chroma: int=12, **kwargs: Any) -> np.ndarray:
    """Compute a chromagram from a waveform or power spectrogram.

    This implementation is derived from ``chromagram_E`` [#]_

    .. [#] Ellis, Daniel P.W.  "Chroma feature analysis and synthesis"
           2007/04/21
           https://www.ee.columbia.edu/~dpwe/resources/matlab/chroma-ansyn/

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        power spectrogram
    norm : float or None
        Column-wise normalization.
        See `librosa.util.normalize` for details.
        If `None`, no normalization is performed.
    n_fft : int  > 0 [scalar]
        FFT window size if provided ``y, sr`` instead of ``S``
    hop_length : int > 0 [scalar]
        hop length if provided ``y, sr`` instead of ``S``
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
    tuning : float [scalar] or None.
        Deviation from A440 tuning in fractional chroma bins.
        If `None`, it is automatically estimated.
    n_chroma : int > 0 [scalar]
        Number of chroma bins to produce (12 by default).
    **kwargs : additional keyword arguments to parameterize chroma filters.
    ctroct : float > 0 [scalar]
    octwidth : float > 0 or None [scalar]
        ``ctroct`` and ``octwidth`` specify a dominance window:
        a Gaussian weighting centered on ``ctroct`` (in octs, A0 = 27.5Hz)
        and with a gaussian half-width of ``octwidth``.
        Set ``octwidth`` to `None` to use a flat weighting.
    norm : float > 0 or np.inf
        Normalization factor for each filter
    base_c : bool
        If True, the filter bank will start at 'C'.
        If False, the filter bank will start at 'A'.
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    chromagram : np.ndarray [shape=(..., n_chroma, t)]
        Normalized energy for each chroma bin at each frame.

    See Also
    --------
    librosa.filters.chroma : Chroma filter bank construction
    librosa.util.normalize : Vector normalization

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> librosa.feature.chroma_stft(y=y, sr=sr)
    array([[1.   , 0.962, ..., 0.143, 0.278],
           [0.688, 0.745, ..., 0.103, 0.162],
           ...,
           [0.468, 0.598, ..., 0.18 , 0.342],
           [0.681, 0.702, ..., 0.553, 1.   ]], dtype=float32)

    Use an energy (magnitude) spectrum instead of power spectrogram

    >>> S = np.abs(librosa.stft(y))
    >>> chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    >>> chroma
    array([[1.   , 0.973, ..., 0.527, 0.569],
           [0.774, 0.81 , ..., 0.518, 0.506],
           ...,
           [0.624, 0.73 , ..., 0.611, 0.644],
           [0.766, 0.822, ..., 0.92 , 1.   ]], dtype=float32)

    Use a pre-computed power spectrogram with a larger frame

    >>> S = np.abs(librosa.stft(y, n_fft=4096))**2
    >>> chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    >>> chroma
    array([[0.994, 0.873, ..., 0.169, 0.227],
           [0.735, 0.64 , ..., 0.141, 0.135],
           ...,
           [0.6  , 0.937, ..., 0.214, 0.257],
           [0.743, 0.937, ..., 0.684, 0.815]], dtype=float32)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax[0])
    >>> fig.colorbar(img, ax=[ax[0]])
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> fig.colorbar(img, ax=[ax[1]])
    """
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, power=2, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)
    chromafb = filters.chroma(sr=sr, n_fft=n_fft, tuning=tuning, n_chroma=n_chroma, **kwargs)
    raw_chroma = np.einsum('cf,...ft->...ct', chromafb, S, optimize=True)
    return util.normalize(raw_chroma, norm=norm, axis=-2)