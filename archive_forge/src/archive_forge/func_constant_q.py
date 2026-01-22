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
def constant_q(*, sr: float, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, bins_per_octave: int=12, window: _WindowSpec='hann', filter_scale: float=1, pad_fft: bool=True, norm: Optional[float]=1, dtype: DTypeLike=np.complex64, gamma: float=0, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a constant-Q basis.

    This function constructs a filter bank similar to Morlet wavelets,
    where complex exponentials are windowed to different lengths
    such that the number of cycles remains fixed for all frequencies.

    By default, a Hann window (rather than the Gaussian window of Morlet wavelets)
    is used, but this can be controlled by the ``window`` parameter.

    Frequencies are spaced geometrically, increasing by a factor of
    ``(2**(1./bins_per_octave))`` at each successive band.

    .. warning:: This function is deprecated as of v0.9 and will be removed in 1.0.
        See `librosa.filters.wavelet`.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    window : string, tuple, number, or function
        Windowing function to apply to filters.

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : boolean
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the ``mode=`` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    gamma : number >= 0
        Bandwidth offset for variable-Q transforms.
        ``gamma=0`` produces a constant-Q filterbank.

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 64-bit (single precision) complex floating point.

    **kwargs : additional keyword arguments
        Arguments to `np.pad()` when ``pad==True``.

    Returns
    -------
    filters : np.ndarray, ``len(filters) == n_bins``
        ``filters[i]`` is ``i``\\ th time-domain CQT basis filter
    lengths : np.ndarray, ``len(lengths) == n_bins``
        The (fractional) length of each filter

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    wavelet
    constant_q_lengths
    librosa.cqt
    librosa.vqt
    librosa.util.normalize

    Examples
    --------
    Use a shorter window for each filter

    >>> basis, lengths = librosa.filters.constant_q(sr=22050, filter_scale=0.5)

    Plot one octave of filters in time and frequency

    >>> import matplotlib.pyplot as plt
    >>> basis, lengths = librosa.filters.constant_q(sr=22050)
    >>> fig, ax = plt.subplots(nrows=2, figsize=(10, 6))
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes[:12])):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     ax[0].plot(i + f_scale.real)
    ...     ax[0].plot(i + f_scale.imag, linestyle=':')
    >>> ax[0].set(yticks=np.arange(len(notes[:12])), yticklabels=notes[:12],
    ...           ylabel='CQ filters',
    ...           title='CQ filters (one octave, time domain)',
    ...           xlabel='Time (samples at 22050 Hz)')
    >>> ax[0].legend(['Real', 'Imaginary'])
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear', y_axis='cqt_note', ax=ax[1])
    >>> ax[1].set(ylabel='CQ filters', title='CQ filter magnitudes (frequency domain)')
    """
    if fmin is None:
        fmin = note_to_hz('C1')
    lengths = constant_q_lengths(sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, window=window, filter_scale=filter_scale, gamma=gamma)
    freqs = fmin * 2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave)
    filters = []
    for ilen, freq in zip(lengths, freqs):
        sig = util.phasor(np.arange(-ilen // 2, ilen // 2, dtype=float) * 2 * np.pi * freq / sr)
        sig = sig * __float_window(window)(len(sig))
        sig = util.normalize(sig, norm=norm)
        filters.append(sig)
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** np.ceil(np.log2(max_len)))
    else:
        max_len = int(np.ceil(max_len))
    filters = np.asarray([util.pad_center(filt, size=max_len, **kwargs) for filt in filters], dtype=dtype)
    return (filters, np.asarray(lengths))