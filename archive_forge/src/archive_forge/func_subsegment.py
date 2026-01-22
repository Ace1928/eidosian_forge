from decorator import decorator
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import sklearn
import sklearn.cluster
import sklearn.feature_extraction
import sklearn.neighbors
from ._cache import cache
from . import util
from .filters import diagonal_filter
from .util.exceptions import ParameterError
from typing import Any, Callable, Optional, TypeVar, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
@cache(level=30)
def subsegment(data: np.ndarray, frames: np.ndarray, *, n_segments: int=4, axis: int=-1) -> np.ndarray:
    """Sub-divide a segmentation by feature clustering.

    Given a set of frame boundaries (``frames``), and a data matrix (``data``),
    each successive interval defined by ``frames`` is partitioned into
    ``n_segments`` by constrained agglomerative clustering.

    .. note::
        If an interval spans fewer than ``n_segments`` frames, then each
        frame becomes a sub-segment.

    Parameters
    ----------
    data : np.ndarray
        Data matrix to use in clustering
    frames : np.ndarray [shape=(n_boundaries,)], dtype=int, non-negative]
        Array of beat or segment boundaries, as provided by
        `librosa.beat.beat_track`,
        `librosa.onset.onset_detect`,
        or `agglomerative`.
    n_segments : int > 0
        Maximum number of frames to sub-divide each interval.
    axis : int
        Axis along which to apply the segmentation.
        By default, the last index (-1) is taken.

    Returns
    -------
    boundaries : np.ndarray [shape=(n_subboundaries,)]
        List of sub-divided segment boundaries

    See Also
    --------
    agglomerative : Temporal segmentation
    librosa.onset.onset_detect : Onset detection
    librosa.beat.beat_track : Beat tracking

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Load audio, detect beat frames, and subdivide in twos by CQT

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
    >>> cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512))
    >>> subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
    >>> subseg_t = librosa.frames_to_time(subseg, sr=sr, hop_length=512)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(librosa.amplitude_to_db(cqt,
    ...                                                  ref=np.max),
    ...                          y_axis='cqt_hz', x_axis='time', ax=ax)
    >>> lims = ax.get_ylim()
    >>> ax.vlines(beat_times, lims[0], lims[1], color='lime', alpha=0.9,
    ...            linewidth=2, label='Beats')
    >>> ax.vlines(subseg_t, lims[0], lims[1], color='linen', linestyle='--',
    ...            linewidth=1.5, alpha=0.5, label='Sub-beats')
    >>> ax.legend()
    >>> ax.set(title='CQT + Beat and sub-beat markers')
    """
    frames = util.fix_frames(frames, x_min=0, x_max=data.shape[axis], pad=True)
    if n_segments < 1:
        raise ParameterError('n_segments must be a positive integer')
    boundaries = []
    idx_slices = [slice(None)] * data.ndim
    for seg_start, seg_end in zip(frames[:-1], frames[1:]):
        idx_slices[axis] = slice(seg_start, seg_end)
        boundaries.extend(seg_start + agglomerative(data[tuple(idx_slices)], min(seg_end - seg_start, n_segments), axis=axis))
    return np.array(boundaries)