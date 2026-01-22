import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def fftfreq(n, d=1.0):
    """Return the FFT sample frequencies.

    Args:
        n (int): Window length.
        d (scalar): Sample spacing.

    Returns:
        cupy.ndarray: Array of length ``n`` containing the sample frequencies.

    .. seealso:: :func:`numpy.fft.fftfreq`
    """
    return cupy.hstack((cupy.arange(0, (n - 1) // 2 + 1, dtype=np.float64), cupy.arange(-(n // 2), 0, dtype=np.float64))) / (n * d)