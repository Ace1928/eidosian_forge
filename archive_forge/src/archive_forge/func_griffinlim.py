import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from numba import jit
from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def griffinlim(S: np.ndarray, *, n_iter: int=32, hop_length: Optional[int]=None, win_length: Optional[int]=None, n_fft: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, length: Optional[int]=None, pad_mode: _PadModeSTFT='constant', momentum: float=0.99, init: Optional[str]='random', random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]=None) -> np.ndarray:
    """Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm.

    Given a short-time Fourier transform magnitude matrix (``S``), the algorithm randomly
    initializes phase estimates, and then alternates forward- and inverse-STFT
    operations. [#]_

    Note that this assumes reconstruction of a real-valued time-domain signal, and
    that ``S`` contains only the non-negative frequencies (as computed by
    `stft`).

    The "fast" GL method [#]_ uses a momentum parameter to accelerate convergence.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    Parameters
    ----------
    S : np.ndarray [shape=(..., n_fft // 2 + 1, t), non-negative]
        An array of short-time Fourier transform magnitudes as produced by
        `stft`.

    n_iter : int > 0
        The number of iterations to run

    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``

    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``

    n_fft : None or int > 0
        The number of samples per frame.
        By default, this will be inferred from the shape of ``S`` as an even number.
        However, if an odd frame length was used, you can explicitly set ``n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`

    center : boolean
        If ``True``, the STFT is assumed to use centered frames.
        If ``False``, the STFT is assumed to use left-aligned frames.

    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is inferred
        to match the precision of the input spectrogram.

    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
        samples.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    momentum : number >= 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method [1]_.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    init : None or 'random' [default]
        If 'random' (the default), then phase values are initialized randomly
        according to ``random_state``.  This is recommended when the input ``S`` is
        a magnitude spectrogram with no initial phase estimates.

        If `None`, then the phase is initialized from ``S``.  This is useful when
        an initial guess for phase can be provided, or when you want to resume
        Griffin-Lim from a previous output.

    random_state : None, int, np.random.RandomState, or np.random.Generator
        If int, random_state is the seed used by the random number generator
        for phase initialization.

        If `np.random.RandomState` or `np.random.Generator` instance, the random number
        generator itself.

        If `None`, defaults to the `np.random.default_rng()` object.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time-domain signal reconstructed from ``S``

    See Also
    --------
    stft
    istft
    magphase
    filters.get_window

    Examples
    --------
    A basic STFT inverse example

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> # Get the magnitude spectrogram
    >>> S = np.abs(librosa.stft(y))
    >>> # Invert using Griffin-Lim
    >>> y_inv = librosa.griffinlim(S)
    >>> # Invert without estimating phase
    >>> y_istft = librosa.istft(S)

    Wave-plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
    >>> ax[0].set(title='Original', xlabel=None)
    >>> ax[0].label_outer()
    >>> librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
    >>> ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
    >>> ax[1].label_outer()
    >>> librosa.display.waveshow(y_istft, sr=sr, color='r', ax=ax[2])
    >>> ax[2].set_title('Magnitude-only istft reconstruction')
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    else:
        raise ParameterError(f'Unsupported random_state={random_state!r}')
    if momentum > 1:
        warnings.warn(f'Griffin-Lim with momentum={momentum} > 1 can be unstable. Proceed with caution!', stacklevel=2)
    elif momentum < 0:
        raise ParameterError(f'griffinlim() called with momentum={momentum} < 0')
    if n_fft is None:
        n_fft = 2 * (S.shape[-2] - 1)
    angles = np.empty(S.shape, dtype=util.dtype_r2c(S.dtype))
    eps = util.tiny(angles)
    if init == 'random':
        angles[:] = util.phasor(2 * np.pi * rng.random(size=S.shape))
    elif init is None:
        angles[:] = 1.0
    else:
        raise ParameterError(f"init={init} must either None or 'random'")
    rebuilt = None
    tprev = None
    inverse = None
    angles *= S
    for _ in range(n_iter):
        inverse = istft(angles, hop_length=hop_length, win_length=win_length, n_fft=n_fft, window=window, center=center, dtype=dtype, length=length, out=inverse)
        rebuilt = stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, out=rebuilt)
        angles[:] = rebuilt
        if tprev is not None:
            angles -= momentum / (1 + momentum) * tprev
        angles /= np.abs(angles) + eps
        angles *= S
        rebuilt, tprev = (tprev, rebuilt)
    return istft(angles, hop_length=hop_length, win_length=win_length, n_fft=n_fft, window=window, center=center, dtype=dtype, length=length, out=inverse)