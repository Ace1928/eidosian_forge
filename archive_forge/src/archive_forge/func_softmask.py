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
def softmask(X: np.ndarray, X_ref: np.ndarray, *, power: float=1, split_zeros: bool=False) -> np.ndarray:
    """Robustly compute a soft-mask operation.

        ``M = X**power / (X**power + X_ref**power)``

    Parameters
    ----------
    X : np.ndarray
        The (non-negative) input array corresponding to the positive mask elements

    X_ref : np.ndarray
        The (non-negative) array of reference or background elements.
        Must have the same shape as ``X``.

    power : number > 0 or np.inf
        If finite, returns the soft mask computed in a numerically stable way

        If infinite, returns a hard (binary) mask equivalent to ``X > X_ref``.
        Note: for hard masks, ties are always broken in favor of ``X_ref`` (``mask=0``).

    split_zeros : bool
        If `True`, entries where ``X`` and ``X_ref`` are both small (close to 0)
        will receive mask values of 0.5.

        Otherwise, the mask is set to 0 for these entries.

    Returns
    -------
    mask : np.ndarray, shape=X.shape
        The output mask array

    Raises
    ------
    ParameterError
        If ``X`` and ``X_ref`` have different shapes.

        If ``X`` or ``X_ref`` are negative anywhere

        If ``power <= 0``

    Examples
    --------
    >>> X = 2 * np.ones((3, 3))
    >>> X_ref = np.vander(np.arange(3.0))
    >>> X
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]])
    >>> X_ref
    array([[ 0.,  0.,  1.],
           [ 1.,  1.,  1.],
           [ 4.,  2.,  1.]])
    >>> librosa.util.softmask(X, X_ref, power=1)
    array([[ 1.   ,  1.   ,  0.667],
           [ 0.667,  0.667,  0.667],
           [ 0.333,  0.5  ,  0.667]])
    >>> librosa.util.softmask(X_ref, X, power=1)
    array([[ 0.   ,  0.   ,  0.333],
           [ 0.333,  0.333,  0.333],
           [ 0.667,  0.5  ,  0.333]])
    >>> librosa.util.softmask(X, X_ref, power=2)
    array([[ 1. ,  1. ,  0.8],
           [ 0.8,  0.8,  0.8],
           [ 0.2,  0.5,  0.8]])
    >>> librosa.util.softmask(X, X_ref, power=4)
    array([[ 1.   ,  1.   ,  0.941],
           [ 0.941,  0.941,  0.941],
           [ 0.059,  0.5  ,  0.941]])
    >>> librosa.util.softmask(X, X_ref, power=100)
    array([[  1.000e+00,   1.000e+00,   1.000e+00],
           [  1.000e+00,   1.000e+00,   1.000e+00],
           [  7.889e-31,   5.000e-01,   1.000e+00]])
    >>> librosa.util.softmask(X, X_ref, power=np.inf)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [False, False,  True]], dtype=bool)
    """
    if X.shape != X_ref.shape:
        raise ParameterError(f'Shape mismatch: {X.shape}!={X_ref.shape}')
    if np.any(X < 0) or np.any(X_ref < 0):
        raise ParameterError('X and X_ref must be non-negative')
    if power <= 0:
        raise ParameterError('power must be strictly positive')
    dtype = X.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32
    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = Z < np.finfo(dtype).tiny
    Z[bad_idx] = 1
    mask: np.ndarray
    if np.isfinite(power):
        mask = (X / Z) ** power
        ref_mask = (X_ref / Z) ** power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        mask = X > X_ref
    return mask