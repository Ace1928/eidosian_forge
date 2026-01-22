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
def lfiltic(b, a, y, x=None):
    """
    Construct initial conditions for lfilter given input and output vectors.

    Given a linear filter (b, a) and initial conditions on the output `y`
    and the input `x`, return the initial conditions on the state vector zi
    which is used by `lfilter` to generate the output given the input.

    Parameters
    ----------
    b : array_like
        Linear filter term.
    a : array_like
        Linear filter term.
    y : array_like
        Initial conditions.

        If ``N = len(a) - 1``, then ``y = {y[-1], y[-2], ..., y[-N]}``.

        If `y` is too short, it is padded with zeros.
    x : array_like, optional
        Initial conditions.

        If ``M = len(b) - 1``, then ``x = {x[-1], x[-2], ..., x[-M]}``.

        If `x` is not given, its initial conditions are assumed zero.

        If `x` is too short, it is padded with zeros.

    Returns
    -------
    zi : ndarray
        The state vector ``zi = {z_0[-1], z_1[-1], ..., z_K-1[-1]}``,
        where ``K = max(M, N)``.

    See Also
    --------
    lfilter, lfilter_zi

    """
    N = np.size(a) - 1
    M = np.size(b) - 1
    K = max(M, N)
    y = np.asarray(y)
    if x is None:
        result_type = np.result_type(np.asarray(b), np.asarray(a), y)
        if result_type.kind in 'bui':
            result_type = np.float64
        x = np.zeros(M, dtype=result_type)
    else:
        x = np.asarray(x)
        result_type = np.result_type(np.asarray(b), np.asarray(a), y, x)
        if result_type.kind in 'bui':
            result_type = np.float64
        x = x.astype(result_type)
        L = np.size(x)
        if L < M:
            x = np.r_[x, np.zeros(M - L)]
    y = y.astype(result_type)
    zi = np.zeros(K, result_type)
    L = np.size(y)
    if L < N:
        y = np.r_[y, np.zeros(N - L)]
    for m in range(M):
        zi[m] = np.sum(b[m + 1:] * x[:M - m], axis=0)
    for m in range(N):
        zi[m] -= np.sum(a[m + 1:] * y[:N - m], axis=0)
    return zi