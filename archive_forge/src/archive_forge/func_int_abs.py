from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def int_abs(arr):
    """Absolute values of array taking care of max negative int values

    Parameters
    ----------
    arr : array-like

    Returns
    -------
    abs_arr : array
        array the same shape as `arr` in which all negative numbers have been
        changed to positive numbers with the magnitude.

    Examples
    --------
    This kind of thing is confusing in base numpy:

    >>> import numpy as np
    >>> np.abs(np.int8(-128))
    -128

    ``int_abs`` fixes that:

    >>> int_abs(np.int8(-128))
    128
    >>> int_abs(np.array([-128, 127], dtype=np.int8))
    array([128, 127], dtype=uint8)
    >>> int_abs(np.array([-128, 127], dtype=np.float32))
    array([128., 127.], dtype=float32)
    """
    arr = np.array(arr, copy=False)
    dt = arr.dtype
    if dt.kind == 'u':
        return arr
    if dt.kind != 'i':
        return np.absolute(arr)
    out = arr.astype(np.dtype(dt.str.replace('i', 'u')))
    return np.choose(arr < 0, (arr, arr * -1), out=out)