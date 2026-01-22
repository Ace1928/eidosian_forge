import numpy as np
import scipy
from ._cache import cache
from . import core
from . import util
from .util.exceptions import ParameterError
from .feature.spectral import melspectrogram
from typing import Any, Callable, Iterable, Optional, Union, Sequence
@cache(level=30)
def onset_strength_multi(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, lag: int=1, max_size: int=1, ref: Optional[np.ndarray]=None, detrend: bool=False, center: bool=True, feature: Optional[Callable]=None, aggregate: Optional[Union[Callable, bool]]=None, channels: Optional[Union[Sequence[int], Sequence[slice]]]=None, **kwargs: Any) -> np.ndarray:
    """Compute a spectral flux onset strength envelope across multiple channels.

    Onset strength for channel ``i`` at time ``t`` is determined by::

        mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram

    n_fft : int > 0 [scalar]
        FFT window size for use in ``feature()`` if ``S`` is not provided.

    hop_length : int > 0 [scalar]
        hop length for use in ``feature()`` if ``S`` is not provided.

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    ref : None or np.ndarray [shape=(d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``

        Must support arguments: ``y, sr, n_fft, hop_length``

    aggregate : function or False
        Aggregation function to use when combining onsets
        at different frequency bins.

        If ``False``, then no aggregation is performed.

        Default: `np.mean`

    channels : list or None
        Array of channel boundaries or slice objects.
        If `None`, then a single channel is generated to span all bands.

    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., n_channels, m)]
        array containing the onset strength envelope for each specified channel

    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided

    See Also
    --------
    onset_strength

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")

    Construct a standard onset function over four sub-bands

    >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
    ...                                                     channels=[0, 32, 64, 96, 128])
    >>> img2 = librosa.display.specshow(onset_subbands, x_axis='time', ax=ax[1])
    >>> ax[1].set(ylabel='Sub-bands', title='Sub-band onset strength')
    >>> fig.colorbar(img2, ax=[ax[1]])
    """
    if feature is None:
        feature = melspectrogram
        kwargs.setdefault('fmax', 0.5 * sr)
    if aggregate is None:
        aggregate = np.mean
    if not util.is_positive_int(lag):
        raise ParameterError(f'lag={lag} must be a positive integer')
    if not util.is_positive_int(max_size):
        raise ParameterError(f'max_size={max_size} must be a positive integer')
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))
        S = core.power_to_db(S)
    S = np.atleast_2d(S)
    if ref is None:
        if max_size == 1:
            ref = S
        else:
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)
    elif ref.shape != S.shape:
        raise ParameterError(f'Reference spectrum shape {ref.shape} must match input spectrum {S.shape}')
    onset_env = S[..., lag:] - ref[..., :-lag]
    onset_env = np.maximum(0.0, onset_env)
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False
    if callable(aggregate):
        onset_env = util.sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=-2)
    pad_width = lag
    if center:
        pad_width += n_fft // (2 * hop_length)
    padding = [(0, 0) for _ in onset_env.shape]
    padding[-1] = (int(pad_width), 0)
    onset_env = np.pad(onset_env, padding, mode='constant')
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)
    if center:
        onset_env = onset_env[..., :S.shape[-1]]
    return onset_env