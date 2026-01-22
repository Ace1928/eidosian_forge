import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal

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
    