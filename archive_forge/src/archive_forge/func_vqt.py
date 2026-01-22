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
def vqt(y: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, intervals: Union[str, Collection[float]]='equal', gamma: Optional[float]=None, bins_per_octave: int=12, tuning: Optional[float]=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', res_type: Optional[str]='soxr_hq', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    """Compute the variable-Q transform of an audio signal.

    This implementation is based on the recursive sub-sampling method
    described by [#]_.

    .. [#] Schörkhuber, Christian, Anssi Klapuri, Nicki Holighaus, and Monika Dörfler.
        "A Matlab toolbox for efficient perfect reconstruction time-frequency
        transforms with log-frequency resolution."
        In Audio Engineering Society Conference: 53rd International Conference: Semantic Audio.
        Audio Engineering Society, 2014.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    hop_length : int > 0 [scalar]
        number of samples between successive VQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at ``fmin``

    intervals : str or array of floats in [1, 2)
        Either a string specification for an interval set, e.g.,
        `'equal'`, `'pythagorean'`, `'ji3'`, etc. or an array of
        intervals expressed as numbers between 1 and 2.
        .. see also:: librosa.interval_frequencies

    gamma : number > 0 [scalar]
        Bandwidth offset for determining filter lengths.

        If ``gamma=0``, produces the constant-Q transform.

        If 'gamma=None', gamma will be calculated such that filter bandwidths are equal to a
        constant fraction of the equivalent rectangular bandwidths (ERB). This is accomplished
        by solving for the gamma which gives::

            B_k = alpha * f_k + gamma = C * ERB(f_k),

        where ``B_k`` is the bandwidth of filter ``k`` with center frequency ``f_k``, alpha
        is the inverse of what would be the constant Q-factor, and ``C = alpha / 0.108`` is the
        constant fraction across all filters.

        Here we use ``ERB(f_k) = 24.7 + 0.108 * f_k``, the best-fit curve derived
        from experimental data in [#]_.

        .. [#] Glasberg, Brian R., and Brian CJ Moore.
            "Derivation of auditory filter shapes from notched-noise data."
            Hearing research 47.1-2 (1990): 103-138.

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float
        Tuning offset in fractions of a bin.

        If ``None``, tuning will be automatically estimated from the signal.

        The minimum frequency of the resulting VQT will be modified to
        ``fmin * 2**(tuning / bins_per_octave)``.

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the VQT basis by discarding up to ``sparsity``
        fraction of the energy in each basis.

        Set ``sparsity=0`` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If ``True``, scale the VQT response by square-root the length of
        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.

        If ``False``, do not scale the VQT. This is analogous to
        ``norm=None`` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.stft` and `numpy.pad`.

    res_type : string
        The resampling mode for recursive downsampling.

    dtype : np.dtype
        The dtype of the output array.  By default, this is inferred to match the
        numerical precision of the input signal.

    Returns
    -------
    VQT : np.ndarray [shape=(..., n_bins, t), dtype=np.complex]
        Variable-Q value each frequency at each time.

    See Also
    --------
    cqt

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Generate and plot a variable-Q power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
    >>> C = np.abs(librosa.cqt(y, sr=sr))
    >>> V = np.abs(librosa.vqt(y, sr=sr))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                          sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[0])
    >>> ax[0].set(title='Constant-Q power spectrum', xlabel=None)
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(V, ref=np.max),
    ...                                sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[1])
    >>> ax[1].set_title('Variable-Q power spectrum')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    if not isinstance(intervals, str):
        bins_per_octave = len(intervals)
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)
    if fmin is None:
        fmin = note_to_hz('C1')
    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)
    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    freqs = interval_frequencies(n_bins=n_bins, fmin=fmin, intervals=intervals, bins_per_octave=bins_per_octave, sort=True)
    freqs_top = freqs[-bins_per_octave:]
    fmax_t: float = np.max(freqs_top)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    lengths, filter_cutoff = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, gamma=gamma, alpha=alpha)
    nyquist = sr / 2.0
    if filter_cutoff > nyquist:
        raise ParameterError(f'Wavelet basis with max frequency={fmax_t} would exceed the Nyquist frequency={nyquist}. Try reducing the number of frequency bins.')
    if res_type is None:
        warnings.warn('Support for VQT with res_type=None is deprecated in librosa 0.10\nand will be removed in version 1.0.', category=FutureWarning, stacklevel=2)
        res_type = 'soxr_hq'
    y, sr, hop_length = __early_downsample(y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale)
    vqt_resp = []
    my_y, my_sr, my_hop = (y, sr, hop_length)
    for i in range(n_octaves):
        if i == 0:
            sl = slice(-n_filters, None)
        else:
            sl = slice(-n_filters * (i + 1), -n_filters * i)
        freqs_oct = freqs[sl]
        alpha_oct = alpha[sl]
        fft_basis, n_fft, _ = __vqt_filter_fft(my_sr, freqs_oct, filter_scale, norm, sparsity, window=window, gamma=gamma, dtype=dtype, alpha=alpha_oct)
        fft_basis[:] *= np.sqrt(sr / my_sr)
        vqt_resp.append(__cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode, dtype=dtype))
        if my_hop % 2 == 0:
            my_hop //= 2
            my_sr /= 2.0
            my_y = audio.resample(my_y, orig_sr=2, target_sr=1, res_type=res_type, scale=True)
    V = __trim_stack(vqt_resp, n_bins, dtype)
    if scale:
        lengths, _ = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, gamma=gamma, alpha=alpha)
        lengths = util.expand_to(lengths, ndim=V.ndim, axes=-2)
        V /= np.sqrt(lengths)
    return V