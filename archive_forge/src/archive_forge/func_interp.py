import cupy
import cupyx.scipy.fft
from cupy import _core
from cupy._core import _routines_math as _math
from cupy._core import fusion
from cupy.lib import stride_tricks
import numpy
def interp(x, xp, fp, left=None, right=None, period=None):
    """ One-dimensional linear interpolation.

    Args:
        x (cupy.ndarray): a 1D array of points on which the interpolation
            is performed.
        xp (cupy.ndarray): a 1D array of points on which the function values
            (``fp``) are known.
        fp (cupy.ndarray): a 1D array containing the function values at the
            the points ``xp``.
        left (float or complex): value to return if ``x < xp[0]``. Default is
            ``fp[0]``.
        right (float or complex): value to return if ``x > xp[-1]``. Default is
            ``fp[-1]``.
        period (None or float): a period for the x-coordinates. Parameters
            ``left`` and ``right`` are ignored if ``period`` is specified.
            Default is ``None``.

    Returns:
        cupy.ndarray: The interpolated values, same shape as ``x``.

    .. note::
        This function may synchronize if ``left`` or ``right`` is not already
        on the device.

    .. seealso:: :func:`numpy.interp`

    """
    if xp.ndim != 1 or fp.ndim != 1:
        raise ValueError('xp and fp must be 1D arrays')
    if xp.size != fp.size:
        raise ValueError('fp and xp are not of the same length')
    if xp.size == 0:
        raise ValueError('array of sample points is empty')
    if not x.flags.c_contiguous:
        raise NotImplementedError('Non-C-contiguous x is currently not supported')
    x_dtype = cupy.common_type(x, xp)
    if not cupy.can_cast(x_dtype, cupy.float64):
        raise TypeError("Cannot cast array data from {} to {} according to the rule 'safe'".format(x_dtype, cupy.float64))
    if period is not None:
        if period == 0:
            raise ValueError('period must be a non-zero value')
        period = abs(period)
        left = None
        right = None
        x = x.astype(cupy.float64)
        xp = xp.astype(cupy.float64)
        x %= period
        xp %= period
        asort_xp = cupy.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = cupy.concatenate((xp[-1:] - period, xp, xp[0:1] + period))
        fp = cupy.concatenate((fp[-1:], fp, fp[0:1]))
        assert xp.flags.c_contiguous
        assert fp.flags.c_contiguous
    out_dtype = 'D' if fp.dtype.kind == 'c' else 'd'
    output = cupy.empty(x.shape, dtype=out_dtype)
    idx = cupy.searchsorted(xp, x, side='right')
    left = fp[0] if left is None else cupy.array(left, fp.dtype)
    right = fp[-1] if right is None else cupy.array(right, fp.dtype)
    kern = _get_interp_kernel(out_dtype == 'D')
    kern(x, idx, xp, fp, xp.size, left, right, output)
    return output