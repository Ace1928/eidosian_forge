import functools
import numpy
import cupy
from cupy._core import _routines_statistics as _statistics
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    """Returns the variance along an axis ignoring NaN values.

    Args:
        a (cupy.ndarray): Array to compute variance.
        axis (int): Along which axis to compute variance. The flattened array
            is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The variance of the input array along the axis.

    .. seealso:: :func:`numpy.nanvar`

    """
    if a.dtype.kind in 'biu':
        return a.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
    return _statistics._nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)