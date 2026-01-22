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
def chroma_cens(*, y: Optional[np.ndarray]=None, sr: float=22050, C: Optional[np.ndarray]=None, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, tuning: Optional[float]=None, n_chroma: int=12, n_octaves: int=7, bins_per_octave: int=36, cqt_mode: str='full', window: Optional[np.ndarray]=None, norm: Optional[float]=2, win_len_smooth: Optional[int]=41, smoothing_window: _WindowSpec='hann') -> np.ndarray:
    """Compute the chroma variant "Chroma Energy Normalized" (CENS)

    To compute CENS features, following steps are taken after obtaining chroma vectors
    using `chroma_cqt`: [#]_.

        1. L-1 normalization of each chroma vector
        2. Quantization of amplitude based on "log-like" amplitude thresholds
        3. (optional) Smoothing with sliding window. Default window length = 41 frames
        4. (not implemented) Downsampling

    CENS features are robust to dynamics, timbre and articulation, thus these are commonly used in audio
    matching and retrieval applications.

    .. [#] Meinard Müller and Sebastian Ewert
           "Chroma Toolbox: MATLAB implementations for extracting variants of chroma-based audio features"
           In Proceedings of the International Conference on Music Information Retrieval (ISMIR), 2011.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time series. Multi-channel is supported.
    sr : number > 0
        sampling rate of ``y``
    C : np.ndarray [shape=(d, t)] [Optional]
        a pre-computed constant-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    tuning : float [scalar] or None.
        Deviation (in fractions of a CQT bin) from A440 tuning
    n_chroma : int > 0
        Number of chroma bins to produce
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`
    bins_per_octave : int > 0
        Number of bins per octave in the CQT.
        Default: 36
    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode
    win_len_smooth : int > 0 or None
        Length of temporal smoothing window. `None` disables temporal smoothing.
        Default: 41
    smoothing_window : str, float or tuple
        Type of window function for temporal smoothing. See `librosa.filters.get_window` for possible inputs.
        Default: 'hann'

    Returns
    -------
    cens : np.ndarray [shape=(..., n_chroma, t)]
        The output cens-chromagram

    See Also
    --------
    chroma_cqt : Compute a chromagram from a constant-Q transform.
    chroma_stft : Compute a chromagram from an STFT spectrogram or waveform.
    librosa.filters.get_window : Compute a window function.

    Examples
    --------
    Compare standard cqt chroma to CENS.

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='chroma_cq')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='chroma_cens')
    >>> fig.colorbar(img, ax=ax)
    """
    if not (win_len_smooth is None or (isinstance(win_len_smooth, (int, np.integer)) and win_len_smooth > 0)):
        raise ParameterError(f'win_len_smooth={win_len_smooth} must be a positive integer or None')
    chroma = chroma_cqt(y=y, C=C, sr=sr, hop_length=hop_length, fmin=fmin, bins_per_octave=bins_per_octave, tuning=tuning, norm=None, n_chroma=n_chroma, n_octaves=n_octaves, cqt_mode=cqt_mode, window=window)
    chroma = util.normalize(chroma, norm=1, axis=-2)
    QUANT_STEPS = [0.4, 0.2, 0.1, 0.05]
    QUANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
    chroma_quant = np.zeros_like(chroma)
    for cur_quant_step_idx, cur_quant_step in enumerate(QUANT_STEPS):
        chroma_quant += (chroma > cur_quant_step) * QUANT_WEIGHTS[cur_quant_step_idx]
    if win_len_smooth:
        win = filters.get_window(smoothing_window, win_len_smooth + 2, fftbins=False)
        win /= np.sum(win)
        win = util.expand_to(win, ndim=chroma_quant.ndim, axes=-1)
        cens = scipy.ndimage.convolve(chroma_quant, win, mode='constant')
    else:
        cens = chroma_quant
    return util.normalize(cens, norm=norm, axis=-2)