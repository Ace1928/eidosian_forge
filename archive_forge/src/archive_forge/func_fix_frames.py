from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
def fix_frames(frames: _SequenceLike[int], *, x_min: Optional[int]=0, x_max: Optional[int]=None, pad: bool=True) -> np.ndarray:
    """Fix a list of frames to lie within [x_min, x_max]

    Examples
    --------
    >>> # Generate a list of frame indices
    >>> frames = np.arange(0, 1000.0, 50)
    >>> frames
    array([   0.,   50.,  100.,  150.,  200.,  250.,  300.,  350.,
            400.,  450.,  500.,  550.,  600.,  650.,  700.,  750.,
            800.,  850.,  900.,  950.])
    >>> # Clip to span at most 250
    >>> librosa.util.fix_frames(frames, x_max=250)
    array([  0,  50, 100, 150, 200, 250])
    >>> # Or pad to span up to 2500
    >>> librosa.util.fix_frames(frames, x_max=2500)
    array([   0,   50,  100,  150,  200,  250,  300,  350,  400,
            450,  500,  550,  600,  650,  700,  750,  800,  850,
            900,  950, 2500])
    >>> librosa.util.fix_frames(frames, x_max=2500, pad=False)
    array([  0,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
           550, 600, 650, 700, 750, 800, 850, 900, 950])

    >>> # Or starting away from zero
    >>> frames = np.arange(200, 500, 33)
    >>> frames
    array([200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames, x_max=500)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497,
           500])

    Parameters
    ----------
    frames : np.ndarray [shape=(n_frames,)]
        List of non-negative frame indices
    x_min : int >= 0 or None
        Minimum allowed frame index
    x_max : int >= 0 or None
        Maximum allowed frame index
    pad : boolean
        If ``True``, then ``frames`` is expanded to span the full range
        ``[x_min, x_max]``

    Returns
    -------
    fixed_frames : np.ndarray [shape=(n_fixed_frames,), dtype=int]
        Fixed frame indices, flattened and sorted

    Raises
    ------
    ParameterError
        If ``frames`` contains negative values
    """
    frames = np.asarray(frames)
    if np.any(frames < 0):
        raise ParameterError('Negative frame index detected')
    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)
    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((np.asarray(pad_data), frames))
    if x_min is not None:
        frames = frames[frames >= x_min]
    if x_max is not None:
        frames = frames[frames <= x_max]
    unique: np.ndarray = np.unique(frames).astype(int)
    return unique