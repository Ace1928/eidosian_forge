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
def reassigned_spectrogram(y: np.ndarray, *, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, reassign_frequencies: bool=True, reassign_times: bool=True, ref_power: Union[float, Callable]=1e-06, fill_nan: bool=False, clip: bool=True, dtype: Optional[DTypeLike]=None, pad_mode: _PadModeSTFT='constant') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-frequency reassigned spectrogram.

    The reassignment vectors are calculated using equations 5.20 and 5.23 in
    [#]_::

        t_reassigned = t + np.real(S_th/S_h)
        omega_reassigned = omega - np.imag(S_dh/S_h)

    where ``S_h`` is the complex STFT calculated using the original window,
    ``S_dh`` is the complex STFT calculated using the derivative of the original
    window, and ``S_th`` is the complex STFT calculated using the original window
    multiplied by the time offset from the window center. See [#]_ for
    additional algorithms, and [#]_ and [#]_ for history and discussion of the
    method.

    .. [#] Flandrin, P., Auger, F., & Chassande-Mottin, E. (2002).
        Time-Frequency reassignment: From principles to algorithms. In
        Applications in Time-Frequency Signal Processing (Vol. 10, pp.
        179-204). CRC Press.

    .. [#] Fulop, S. A., & Fitz, K. (2006). Algorithms for computing the
        time-corrected instantaneous frequency (reassigned) spectrogram, with
        applications. The Journal of the Acoustical Society of America, 119(1),
        360. doi:10.1121/1.2133000

    .. [#] Auger, F., Flandrin, P., Lin, Y.-T., McLaughlin, S., Meignen, S.,
        Oberlin, T., & Wu, H.-T. (2013). Time-Frequency Reassignment and
        Synchrosqueezing: An Overview. IEEE Signal Processing Magazine, 30(6),
        32-41. doi:10.1109/MSP.2013.2265316

    .. [#] Hainsworth, S., Macleod, M. (2003). Time-frequency reassignment: a
        review and analysis. Tech. Rep. CUED/FINFENG/TR.459, Cambridge
        University Engineering Department

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to ``reassigned_spectrogram``

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        See `stft` for details.

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True`` (default), the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``. See `Notes` for
          recommended usage in this function.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    reassign_frequencies : boolean
        - If ``True`` (default), the returned frequencies will be instantaneous
          frequency estimates.
        - If ``False``, the returned frequencies will be a read-only view of the
          STFT bin frequencies for all frames.

    reassign_times : boolean
        - If ``True`` (default), the returned times will be corrected
          (reassigned) time estimates for each bin.
        - If ``False``, the returned times will be a read-only view of the STFT
          frame times for all bins.

    ref_power : float >= 0 or callable
        Minimum power threshold for estimating time-frequency reassignments.
        Any bin with ``np.abs(S[f, t])**2 < ref_power`` will be returned as
        `np.nan` in both frequency and time, unless ``fill_nan`` is ``True``. If 0
        is provided, then only bins with zero power will be returned as
        `np.nan` (unless ``fill_nan=True``).

    fill_nan : boolean
        - If ``False`` (default), the frequency and time reassignments for bins
          below the power threshold provided in ``ref_power`` will be returned as
          `np.nan`.
        - If ``True``, the frequency and time reassignments for these bins will
          be returned as the bin center frequencies and frame times.

    clip : boolean
        - If ``True`` (default), estimated frequencies outside the range
          `[0, 0.5 * sr]` or times outside the range `[0, len(y) / sr]` will be
          clipped to those ranges.
        - If ``False``, estimated frequencies and times beyond the bounds of the
          spectrogram may be returned.

    dtype : numeric type
        Complex numeric type for STFT calculation. Default is inferred to match
        the precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    freqs, times, mags : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Instantaneous frequencies:
            ``freqs[..., f, t]`` is the frequency for bin ``f``, frame ``t``.
            If ``reassign_frequencies=False``, this will instead be a read-only array
            of the same shape containing the bin center frequencies for all frames.

        Reassigned times:
            ``times[..., f, t]`` is the time for bin ``f``, frame ``t``.
            If ``reassign_times=False``, this will instead be a read-only array of
            the same shape containing the frame times for all bins.

        Magnitudes from short-time Fourier transform:
            ``mags[..., f, t]`` is the magnitude for bin ``f``, frame ``t``.

    Warns
    -----
    RuntimeWarning
        Frequency or time estimates with zero support will produce a
        divide-by-zero warning, and will be returned as `np.nan` unless
        ``fill_nan=True``.

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    It is recommended to use ``center=False`` with this function rather than the
    librosa default ``True``. Unlike ``stft``, reassigned times are not aligned to
    the left or center of each frame, so padding the signal does not affect the
    meaning of the reassigned times. However, reassignment assumes that the
    energy in each FFT bin is associated with exactly one signal component and
    impulse event.

    If ``reassign_times`` is ``False``, the frame times that are returned will be
    aligned to the left or center of the frame, depending on the value of
    ``center``. In this case, if ``center`` is ``True``, then ``pad_mode="wrap"`` is
    recommended for valid estimation of the instantaneous frequencies in the
    boundary frames.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> amin = 1e-10
    >>> n_fft = 64
    >>> sr = 4000
    >>> y = 1e-3 * librosa.clicks(times=[0.3], sr=sr, click_duration=1.0,
    ...                           click_freq=1200.0, length=8000) +\\
    ...     1e-3 * librosa.clicks(times=[1.5], sr=sr, click_duration=0.5,
    ...                           click_freq=400.0, length=8000) +\\
    ...     1e-3 * librosa.chirp(fmin=200, fmax=1600, sr=sr, duration=2.0) +\\
    ...     1e-6 * np.random.randn(2*sr)
    >>> freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=sr,
    ...                                                     n_fft=n_fft)
    >>> mags_db = librosa.amplitude_to_db(mags, ref=np.max)

    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear", sr=sr,
    ...                          hop_length=n_fft//4, ax=ax[0])
    >>> ax[0].set(title="Spectrogram", xlabel=None)
    >>> ax[0].label_outer()
    >>> ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
    >>> ax[1].set_title("Reassigned spectrogram")
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """
    if not callable(ref_power) and ref_power < 0:
        raise ParameterError('ref_power must be non-negative or callable.')
    if not reassign_frequencies and (not reassign_times):
        raise ParameterError('reassign_frequencies or reassign_times must be True.')
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    if reassign_frequencies:
        freqs, S = __reassign_frequencies(y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    if reassign_times:
        times, S = __reassign_times(y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    assert S is not None
    mags: np.ndarray = np.abs(S)
    if fill_nan or not reassign_frequencies or (not reassign_times):
        if center:
            pad_length = None
        else:
            pad_length = n_fft
        bin_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
        frame_times = convert.frames_to_time(frames=np.arange(S.shape[-1]), sr=sr, hop_length=hop_length, n_fft=pad_length)
    if callable(ref_power):
        ref_p = ref_power(mags ** 2)
    else:
        ref_p = ref_power
    mags_low = np.less(mags, ref_p ** 0.5, where=~np.isnan(mags))
    if reassign_frequencies:
        if ref_p > 0:
            freqs[mags_low] = np.nan
        if fill_nan:
            freqs = np.where(np.isnan(freqs), bin_freqs[:, np.newaxis], freqs)
        if clip:
            np.clip(freqs, 0, sr / 2.0, out=freqs)
    else:
        freqs = np.broadcast_to(bin_freqs[:, np.newaxis], S.shape)
    if reassign_times:
        if ref_p > 0:
            times[mags_low] = np.nan
        if fill_nan:
            times = np.where(np.isnan(times), frame_times[np.newaxis, :], times)
        if clip:
            np.clip(times, 0, y.shape[-1] / float(sr), out=times)
    else:
        times = np.broadcast_to(frame_times[np.newaxis, :], S.shape)
    return (freqs, times, mags)