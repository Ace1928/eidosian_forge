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
@cache(level=20)
def hybrid_cqt(y: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, bins_per_octave: int=12, tuning: Optional[float]=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', res_type: str='soxr_hq', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    """Compute the hybrid constant-Q transform of an audio signal.

    Here, the hybrid CQT uses the pseudo CQT for higher frequencies where
    the hop_length is longer than half the filter length and the full CQT
    for lower frequencies.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at ``fmin``

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If ``None``, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting CQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

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
        If ``True``, scale the CQT response by square-root the length of
        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the CQT. This is analogous to
        ``norm=None`` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    res_type : string
        Resampling mode.  See `librosa.cqt` for details.

    dtype : np.dtype, optional
        The complex dtype to use for computing the CQT.
        By default, this is inferred to match the precision of
        the input signal.

    Returns
    -------
    CQT : np.ndarray [shape=(..., n_bins, t), dtype=np.float]
        Constant-Q energy for each frequency at each time.

    See Also
    --------
    cqt
    pseudo_cqt

    Notes
    -----
    This function caches at level 20.
    """
    if fmin is None:
        fmin = note_to_hz('C1')
    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    freqs = cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    lengths, _ = filters.wavelet_lengths(freqs=freqs, sr=sr, filter_scale=filter_scale, window=window, alpha=alpha)
    pseudo_filters = 2.0 ** np.ceil(np.log2(lengths)) < 2 * hop_length
    n_bins_pseudo = int(np.sum(pseudo_filters))
    n_bins_full = n_bins - n_bins_pseudo
    cqt_resp = []
    if n_bins_pseudo > 0:
        fmin_pseudo = np.min(freqs[pseudo_filters])
        cqt_resp.append(pseudo_cqt(y, sr=sr, hop_length=hop_length, fmin=fmin_pseudo, n_bins=n_bins_pseudo, bins_per_octave=bins_per_octave, filter_scale=filter_scale, norm=norm, sparsity=sparsity, window=window, scale=scale, pad_mode=pad_mode, dtype=dtype))
    if n_bins_full > 0:
        cqt_resp.append(np.abs(cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins_full, bins_per_octave=bins_per_octave, filter_scale=filter_scale, norm=norm, sparsity=sparsity, window=window, scale=scale, pad_mode=pad_mode, res_type=res_type, dtype=dtype)))
    return __trim_stack(cqt_resp, n_bins, cqt_resp[-1].dtype)