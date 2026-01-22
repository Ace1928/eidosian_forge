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
@cache(level=40)
def icqt(C: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, bins_per_octave: int=12, tuning: float=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, length: Optional[int]=None, res_type: str='soxr_hq', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    """Compute the inverse constant-Q transform.

    Given a constant-Q transform representation ``C`` of an audio signal ``y``,
    this function produces an approximation ``y_hat``.

    Parameters
    ----------
    C : np.ndarray, [shape=(..., n_bins, n_frames)]
        Constant-Q representation as produced by `cqt`

    sr : number > 0 [scalar]
        sampling rate of the signal

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float [scalar]
        Tuning offset in fractions of a bin.

        The minimum frequency of the CQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0 [scalar]
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the CQT response by square-root the length
        of each channel's filter. This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the CQT. This is analogous to ``norm=None``
        in FFT.

    length : int > 0, optional
        If provided, the output ``y`` is zero-padded or clipped to exactly
        ``length`` samples.

    res_type : string
        Resampling mode.
        See `librosa.resample` for supported modes.

    dtype : numeric type
        Real numeric type for ``y``.  Default is inferred to match the numerical
        precision of the input CQT.

    Returns
    -------
    y : np.ndarray, [shape=(..., n_samples), dtype=np.float]
        Audio time-series reconstructed from the CQT representation.

    See Also
    --------
    cqt
    librosa.resample

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    Using default parameters

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> C = librosa.cqt(y=y, sr=sr)
    >>> y_hat = librosa.icqt(C=C, sr=sr)

    Or with a different hop length and frequency resolution:

    >>> hop_length = 256
    >>> bins_per_octave = 12 * 3
    >>> C = librosa.cqt(y=y, sr=sr, hop_length=256, n_bins=7*bins_per_octave,
    ...                 bins_per_octave=bins_per_octave)
    >>> y_hat = librosa.icqt(C=C, sr=sr, hop_length=hop_length,
    ...                 bins_per_octave=bins_per_octave)
    """
    if fmin is None:
        fmin = note_to_hz('C1')
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    n_bins = C.shape[-2]
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    freqs = cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    lengths, f_cutoff = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, alpha=alpha)
    if length is not None:
        n_frames = int(np.ceil((length + max(lengths)) / hop_length))
        C = C[..., :n_frames]
    C_scale = np.sqrt(lengths)
    y: Optional[np.ndarray] = None
    srs = [sr]
    hops = [hop_length]
    for i in range(n_octaves - 1):
        if hops[0] % 2 == 0:
            srs.insert(0, srs[0] * 0.5)
            hops.insert(0, hops[0] // 2)
        else:
            srs.insert(0, srs[0])
            hops.insert(0, hops[0])
    for i, (my_sr, my_hop) in enumerate(zip(srs, hops)):
        n_filters = min(bins_per_octave, n_bins - bins_per_octave * i)
        sl = slice(bins_per_octave * i, bins_per_octave * i + n_filters)
        fft_basis, n_fft, _ = __vqt_filter_fft(my_sr, freqs[sl], filter_scale, norm, sparsity, window=window, dtype=dtype, alpha=alpha[sl])
        inv_basis = fft_basis.H.todense()
        freq_power = 1 / np.sum(util.abs2(np.asarray(inv_basis)), axis=0)
        freq_power *= n_fft / lengths[sl]
        if scale:
            D_oct = np.einsum('fc,c,c,...ct->...ft', inv_basis, C_scale[sl], freq_power, C[..., sl, :], optimize=True)
        else:
            D_oct = np.einsum('fc,c,...ct->...ft', inv_basis, freq_power, C[..., sl, :], optimize=True)
        y_oct = istft(D_oct, window='ones', hop_length=my_hop, dtype=dtype)
        y_oct = audio.resample(y_oct, orig_sr=1, target_sr=sr // my_sr, res_type=res_type, scale=False, fix=False)
        if y is None:
            y = y_oct
        else:
            y[..., :y_oct.shape[-1]] += y_oct
    assert y is not None
    if length:
        y = util.fix_length(y, size=length)
    return y