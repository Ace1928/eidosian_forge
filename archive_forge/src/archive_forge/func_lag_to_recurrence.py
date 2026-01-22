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
def lag_to_recurrence(lag: _ArrayOrSparseMatrix, *, axis: int=-1) -> _ArrayOrSparseMatrix:
    """Convert a lag matrix into a recurrence matrix.

    Parameters
    ----------
    lag : np.ndarray or scipy.sparse.spmatrix
        A lag matrix, as produced by ``recurrence_to_lag``
    axis : int
        The axis corresponding to the time dimension.
        The alternate axis will be interpreted in lag coordinates.

    Returns
    -------
    rec : np.ndarray or scipy.sparse.spmatrix [shape=(n, n)]
        A recurrence matrix in (time, time) coordinates
        For sparse matrices, format will match that of ``lag``.

    Raises
    ------
    ParameterError : if ``lag`` does not have the correct shape

    See Also
    --------
    recurrence_to_lag

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 1024
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    >>> recurrence = librosa.segment.recurrence_matrix(chroma_stack)
    >>> lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
    >>> lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)
    >>> rec_pad = librosa.segment.lag_to_recurrence(lag_pad)
    >>> rec_nopad = librosa.segment.lag_to_recurrence(lag_nopad)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    >>> librosa.display.specshow(lag_pad, x_axis='s', y_axis='lag',
    ...                          hop_length=hop_length, ax=ax[0, 0])
    >>> ax[0, 0].set(title='Lag (zero-padded)')
    >>> ax[0, 0].label_outer()
    >>> librosa.display.specshow(lag_nopad, x_axis='s', y_axis='time',
    ...                          hop_length=hop_length, ax=ax[0, 1])
    >>> ax[0, 1].set(title='Lag (no padding)')
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(rec_pad, x_axis='s', y_axis='time',
    ...                          hop_length=hop_length, ax=ax[1, 0])
    >>> ax[1, 0].set(title='Recurrence (with padding)')
    >>> librosa.display.specshow(rec_nopad, x_axis='s', y_axis='time',
    ...                          hop_length=hop_length, ax=ax[1, 1])
    >>> ax[1, 1].set(title='Recurrence (without padding)')
    >>> ax[1, 1].label_outer()
    """
    if axis not in [0, 1, -1]:
        raise ParameterError(f'Invalid target axis: {axis}')
    axis = np.abs(axis)
    if lag.ndim != 2 or (lag.shape[0] != lag.shape[1] and lag.shape[1 - axis] != 2 * lag.shape[axis]):
        raise ParameterError(f'Invalid lag matrix shape: {lag.shape}')
    t = lag.shape[axis]
    rec = util.shear(lag, factor=+1, axis=axis)
    sub_slice = [slice(None)] * rec.ndim
    sub_slice[1 - axis] = slice(t)
    rec_slice: _ArrayOrSparseMatrix = rec[tuple(sub_slice)]
    return rec_slice