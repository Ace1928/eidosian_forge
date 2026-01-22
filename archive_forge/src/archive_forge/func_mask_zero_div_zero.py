from numpy in the following ways:
from __future__ import annotations
import operator
import numpy as np
from pandas.core import roperator
def mask_zero_div_zero(x, y, result: np.ndarray) -> np.ndarray:
    """
    Set results of  0 // 0 to np.nan, regardless of the dtypes
    of the numerator or the denominator.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    result : ndarray

    Returns
    -------
    ndarray
        The filled result.

    Examples
    --------
    >>> x = np.array([1, 0, -1], dtype=np.int64)
    >>> x
    array([ 1,  0, -1])
    >>> y = 0       # int 0; numpy behavior is different with float
    >>> result = x // y
    >>> result      # raw numpy result does not fill division by zero
    array([0, 0, 0])
    >>> mask_zero_div_zero(x, y, result)
    array([ inf,  nan, -inf])
    """
    if not hasattr(y, 'dtype'):
        y = np.array(y)
    if not hasattr(x, 'dtype'):
        x = np.array(x)
    zmask = y == 0
    if zmask.any():
        zneg_mask = zmask & np.signbit(y)
        zpos_mask = zmask & ~zneg_mask
        x_lt0 = x < 0
        x_gt0 = x > 0
        nan_mask = zmask & (x == 0)
        neginf_mask = zpos_mask & x_lt0 | zneg_mask & x_gt0
        posinf_mask = zpos_mask & x_gt0 | zneg_mask & x_lt0
        if nan_mask.any() or neginf_mask.any() or posinf_mask.any():
            result = result.astype('float64', copy=False)
            result[nan_mask] = np.nan
            result[posinf_mask] = np.inf
            result[neginf_mask] = -np.inf
    return result