import warnings
import numpy as np
from numba import jit
from . import audio
from .intervals import interval_frequencies
from .fft import get_fftlib
from .convert import cqt_frequencies, note_to_hz
from .spectrum import stft, istft
from .pitch import estimate_tuning
from .._cache import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError
from numpy.typing import DTypeLike
from typing import Optional, Union, Collection, List
from .._typing import _WindowSpec, _PadMode, _FloatLike_co, _ensure_not_reachable
def griffinlim_cqt(C: np.ndarray, *, n_iter: int=32, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, bins_per_octave: int=12, tuning: float=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', res_type: str='soxr_hq', dtype: Optional[DTypeLike]=None, length: Optional[int]=None, momentum: float=0.99, init: Optional[str]='random', random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]=None) -> np.ndarray:
    """Approximate constant-Q magnitude spectrogram inversion using the "fast" Griffin-Lim
    algorithm.

    Given the magnitude of a constant-Q spectrogram (``C``), the algorithm randomly initializes
    phase estimates, and then alternates forward- and inverse-CQT operations. [#]_

    This implementation is based on the (fast) Griffin-Lim method for Short-time Fourier Transforms, [#]_
    but adapted for use with constant-Q spectrograms.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    Parameters
    ----------
    C : np.ndarray [shape=(..., n_bins, n_frames)]
        The constant-Q magnitude spectrogram

    n_iter : int > 0
        The number of iterations to run

    sr : number > 0
        Audio sampling rate

    hop_length : int > 0
        The hop length of the CQT

    fmin : number > 0
        Minimum frequency for the CQT.

        If not provided, it defaults to `C1`.

    bins_per_octave : int > 0
        Number of bins per octave

    tuning : float
        Tuning deviation from A440, in fractions of a bin

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length
        of each channel's filter.  This is analogous to ``norm='ortho'``
        in FFT.

        If ``False``, do not scale the CQT. This is analogous to ``norm=None``
        in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    res_type : string
        The resampling mode for recursive downsampling.

        See ``librosa.resample`` for a list of available options.

    dtype : numeric type
        Real numeric type for ``y``.  Default is inferred to match the precision
        of the input CQT.

    length : int > 0, optional
        If provided, the output ``y`` is zero-padded or clipped to exactly
        ``length`` samples.

    momentum : float > 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    init : None or 'random' [default]
        If 'random' (the default), then phase values are initialized randomly
        according to ``random_state``.  This is recommended when the input ``C`` is
        a magnitude spectrogram with no initial phase estimates.

        If ``None``, then the phase is initialized from ``C``.  This is useful when
        an initial guess for phase can be provided, or when you want to resume
        Griffin-Lim from a previous output.

    random_state : None, int, np.random.RandomState, or np.random.Generator
        If int, random_state is the seed used by the random number generator
        for phase initialization.

        If `np.random.RandomState` or `np.random.Generator` instance, the random number generator itself.

        If ``None``, defaults to the `np.random.default_rng()` object.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time-domain signal reconstructed from ``C``

    See Also
    --------
    cqt
    icqt
    griffinlim
    filters.get_window
    resample

    Examples
    --------
    A basis CQT inverse example

    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), sr=None)
    >>> # Get the CQT magnitude, 7 octaves at 36 bins per octave
    >>> C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=36, n_bins=7*36))
    >>> # Invert using Griffin-Lim
    >>> y_inv = librosa.griffinlim_cqt(C, sr=sr, bins_per_octave=36)
    >>> # And invert without estimating phase
    >>> y_icqt = librosa.icqt(C, sr=sr, bins_per_octave=36)

    Wave-plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
    >>> ax[0].set(title='Original', xlabel=None)
    >>> ax[0].label_outer()
    >>> librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
    >>> ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
    >>> ax[1].label_outer()
    >>> librosa.display.waveshow(y_icqt, sr=sr, color='r', ax=ax[2])
    >>> ax[2].set(title='Magnitude-only icqt reconstruction')
    """
    if fmin is None:
        fmin = note_to_hz('C1')
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    else:
        _ensure_not_reachable(random_state)
        raise ParameterError(f'Unsupported random_state={random_state!r}')
    if momentum > 1:
        warnings.warn(f'Griffin-Lim with momentum={momentum} > 1 can be unstable. Proceed with caution!', stacklevel=2)
    elif momentum < 0:
        raise ParameterError(f'griffinlim_cqt() called with momentum={momentum} < 0')
    angles = np.empty(C.shape, dtype=np.complex64)
    eps = util.tiny(angles)
    if init == 'random':
        angles[:] = util.phasor(2 * np.pi * rng.random(size=C.shape))
    elif init is None:
        angles[:] = 1.0
    else:
        raise ParameterError(f"init={init} must either None or 'random'")
    rebuilt: np.ndarray = np.array(0.0)
    for _ in range(n_iter):
        tprev = rebuilt
        inverse = icqt(C * angles, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, fmin=fmin, tuning=tuning, filter_scale=filter_scale, window=window, length=length, res_type=res_type, norm=norm, scale=scale, sparsity=sparsity, dtype=dtype)
        rebuilt = cqt(inverse, sr=sr, bins_per_octave=bins_per_octave, n_bins=C.shape[-2], hop_length=hop_length, fmin=fmin, tuning=tuning, filter_scale=filter_scale, window=window, norm=norm, scale=scale, sparsity=sparsity, pad_mode=pad_mode, res_type=res_type)
        angles[:] = rebuilt - momentum / (1 + momentum) * tprev
        angles[:] /= np.abs(angles) + eps
    return icqt(C * angles, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=filter_scale, fmin=fmin, window=window, length=length, res_type=res_type, norm=norm, scale=scale, sparsity=sparsity, dtype=dtype)