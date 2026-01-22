import numpy as np
import scipy
from ._cache import cache
from . import core
from . import util
from .util.exceptions import ParameterError
from .feature.spectral import melspectrogram
from typing import Any, Callable, Iterable, Optional, Union, Sequence
def onset_detect(*, y: Optional[np.ndarray]=None, sr: float=22050, onset_envelope: Optional[np.ndarray]=None, hop_length: int=512, backtrack: bool=False, energy: Optional[np.ndarray]=None, units: str='frames', normalize: bool=True, **kwargs: Any) -> np.ndarray:
    """Locate note onset events by picking peaks in an onset strength envelope.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [#]_.

    .. [#] https://github.com/CPJKU/onset_db

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series, must be monophonic

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(m,)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    units : {'frames', 'samples', 'time'}
        The units to encode detected onset events in.
        By default, 'frames' are used.

    backtrack : bool
        If ``True``, detected onset events are backtracked to the nearest
        preceding minimum of ``energy``.

        This is primarily useful when using onsets as slice points for segmentation.

    energy : np.ndarray [shape=(m,)] (optional)
        An energy function to use for backtracking detected onset events.
        If none is provided, then ``onset_envelope`` is used.

    normalize : bool
        If ``True`` (default), normalize the onset envelope to have minimum of 0 and
        maximum of 1 prior to detection.  This is helpful for standardizing the
        parameters of `librosa.util.peak_pick`.

        Otherwise, the onset envelope is left unnormalized.

    **kwargs : additional keyword arguments
        Additional parameters for peak picking.

        See `librosa.util.peak_pick` for details.

    Returns
    -------
    onsets : np.ndarray [shape=(n_onsets,)]
        estimated positions of detected onsets, in whichever units
        are specified.  By default, frame indices.

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty list.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onsets`` are provided

        or if ``units`` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    onset_strength : compute onset strength per-frame
    onset_backtrack : backtracking onset events
    librosa.util.peak_pick : pick peaks from a time series

    Examples
    --------
    Get onset times from a signal

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.onset.onset_detect(y=y, sr=sr, units='time')
    array([0.07 , 0.232, 0.395, 0.604, 0.743, 0.929, 1.045, 1.115,
           1.416, 1.672, 1.881, 2.043, 2.206, 2.368, 2.554, 3.019])

    Or use a pre-computed onset envelope

    >>> o_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(o_env, sr=sr)
    >>> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          x_axis='time', y_axis='log', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, o_env, label='Onset strength')
    >>> ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
    ...            linestyle='--', label='Onsets')
    >>> ax[1].legend()
    """
    if onset_envelope is None:
        if y is None:
            raise ParameterError('y or onset_envelope must be provided')
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)
    if normalize:
        onset_envelope = onset_envelope - np.min(onset_envelope)
        onset_envelope /= np.max(onset_envelope) + util.tiny(onset_envelope)
    assert onset_envelope is not None
    if not onset_envelope.any() or not np.all(np.isfinite(onset_envelope)):
        onsets = np.array([], dtype=int)
    else:
        kwargs.setdefault('pre_max', 0.03 * sr // hop_length)
        kwargs.setdefault('post_max', 0.0 * sr // hop_length + 1)
        kwargs.setdefault('pre_avg', 0.1 * sr // hop_length)
        kwargs.setdefault('post_avg', 0.1 * sr // hop_length + 1)
        kwargs.setdefault('wait', 0.03 * sr // hop_length)
        kwargs.setdefault('delta', 0.07)
        onsets = util.peak_pick(onset_envelope, **kwargs)
        if backtrack:
            if energy is None:
                energy = onset_envelope
            assert energy is not None
            onsets = onset_backtrack(onsets, energy)
    if units == 'frames':
        pass
    elif units == 'samples':
        onsets = core.frames_to_samples(onsets, hop_length=hop_length)
    elif units == 'time':
        onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError(f'Invalid unit type: {units}')
    return onsets