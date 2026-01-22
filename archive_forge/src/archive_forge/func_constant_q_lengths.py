import warnings
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
from numba import jit
from ._cache import cache
from . import util
from .util.exceptions import ParameterError
from .util.decorators import deprecated
from .core.convert import note_to_hz, hz_to_midi, midi_to_hz, hz_to_octs
from .core.convert import fft_frequencies, mel_frequencies
from numpy.typing import ArrayLike, DTypeLike
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
@deprecated(version='0.9.0', version_removed='1.0')
@cache(level=10)
def constant_q_lengths(*, sr: float, fmin: _FloatLike_co, n_bins: int=84, bins_per_octave: int=12, window: _WindowSpec='hann', filter_scale: float=1, gamma: float=0) -> np.ndarray:
    """Return length of each filter in a constant-Q basis.

    .. warning:: This function is deprecated as of v0.9 and will be removed in 1.0.
        See `librosa.filters.wavelet_lengths`.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    fmin : float > 0 [scalar]
        Minimum frequency bin.
    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).
    bins_per_octave : int > 0 [scalar]
        Number of bins per octave
    window : str or callable
        Window function to use on filters
    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.
    gamma : number >= 0
        Bandwidth offset for variable-Q transforms.
        ``gamma=0`` produces a constant-Q filterbank.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    wavelet_lengths
    """
    if fmin <= 0:
        raise ParameterError('fmin must be strictly positive')
    if bins_per_octave <= 0:
        raise ParameterError('bins_per_octave must be positive')
    if filter_scale <= 0:
        raise ParameterError('filter_scale must be positive')
    if n_bins <= 0 or not isinstance(n_bins, (int, np.integer)):
        raise ParameterError('n_bins must be a positive integer')
    freq = fmin * 2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave)
    alpha = (2.0 ** (2 / bins_per_octave) - 1) / (2.0 ** (2 / bins_per_octave) + 1)
    Q = float(filter_scale) / alpha
    if max(freq * (1 + 0.5 * window_bandwidth(window) / Q)) > sr / 2.0:
        raise ParameterError(f'Maximum filter frequency={max(freq):.2f} would exceed Nyquist={sr / 2}')
    lengths: np.ndarray = Q * sr / (freq + gamma / alpha)
    return lengths