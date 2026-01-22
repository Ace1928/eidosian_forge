import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal
def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along the given axis.

    Args:
        y (cupy.ndarray): Input array to integrate.
        x (cupy.ndarray): Sample points over which to integrate. If None equal
            spacing `dx` is assumed.
        dx (float): Spacing between sample points, used if `x` is None, default
            is 1.
        axis (int): The axis along which the integral is taken, default is
            the last axis.

    Returns:
        cupy.ndarray: Definite integral as approximated by the trapezoidal
        rule.

    .. seealso:: :func:`numpy.trapz`
    """
    if not isinstance(y, cupy.ndarray):
        raise TypeError('`y` should be of type cupy.ndarray')
    if x is None:
        d = dx
    else:
        if not isinstance(x, cupy.ndarray):
            raise TypeError('`x` should be of type cupy.ndarray')
        if x.ndim == 1:
            d = diff(x)
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = cupy.add.reduce(product, axis)
    return ret