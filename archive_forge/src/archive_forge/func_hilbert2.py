import operator
import math
from math import prod as _prod
import timeit
import warnings
from scipy.spatial import cKDTree
from . import _sigtools
from ._ltisys import dlti
from ._upfirdn import upfirdn, _output_len, _upfirdn_modes
from scipy import linalg, fft as sp_fft
from scipy import ndimage
from scipy.fft._helper import _init_nd_shape_and_axes
import numpy as np
from scipy.special import lambertw
from .windows import get_window
from ._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from ._filter_design import cheby1, _validate_sos, zpk2sos
from ._fir_filter_design import firwin
from ._sosfilt import _sosfilt
def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal

    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError('x must be 2-D.')
    if np.iscomplexobj(x):
        raise ValueError('x must be real.')
    if N is None:
        N = x.shape
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError('N must be positive.')
        N = (N, N)
    elif len(N) != 2 or np.any(np.asarray(N) <= 0):
        raise ValueError('When given as a tuple, N must hold exactly two positive integers')
    Xf = sp_fft.fft2(x, N, axes=(0, 1))
    h1 = np.zeros(N[0], dtype=Xf.dtype)
    h2 = np.zeros(N[1], dtype=Xf.dtype)
    for h in (h1, h2):
        N1 = h.shape[0]
        if N1 % 2 == 0:
            h[0] = h[N1 // 2] = 1
            h[1:N1 // 2] = 2
        else:
            h[0] = 1
            h[1:(N1 + 1) // 2] = 2
    h = h1[:, np.newaxis] * h2[np.newaxis, :]
    k = x.ndim
    while k > 2:
        h = h[:, np.newaxis]
        k -= 1
    x = sp_fft.ifft2(Xf * h, axes=(0, 1))
    return x