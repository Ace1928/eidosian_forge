import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import fft, pi
def original_fftshift(x, axes=None):
    """ How fftshift was implemented in v1.14"""
    tmp = asarray(x)
    ndim = tmp.ndim
    if axes is None:
        axes = list(range(ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    y = tmp
    for k in axes:
        n = tmp.shape[k]
        p2 = (n + 1) // 2
        mylist = concatenate((arange(p2, n), arange(p2)))
        y = take(y, mylist, k)
    return y