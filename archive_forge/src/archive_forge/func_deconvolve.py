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
def deconvolve(signal, divisor):
    """Deconvolves ``divisor`` out of ``signal`` using inverse filtering.

    Returns the quotient and remainder such that
    ``signal = convolve(divisor, quotient) + remainder``

    Parameters
    ----------
    signal : (N,) array_like
        Signal data, typically a recorded signal
    divisor : (N,) array_like
        Divisor data, typically an impulse response or filter that was
        applied to the original signal

    Returns
    -------
    quotient : ndarray
        Quotient, typically the recovered original signal
    remainder : ndarray
        Remainder

    See Also
    --------
    numpy.polydiv : performs polynomial division (same operation, but
                    also accepts poly1d objects)

    Examples
    --------
    Deconvolve a signal that's been filtered:

    >>> from scipy import signal
    >>> original = [0, 1, 0, 0, 1, 1, 0, 0]
    >>> impulse_response = [2, 1]
    >>> recorded = signal.convolve(impulse_response, original)
    >>> recorded
    array([0, 2, 1, 0, 2, 3, 1, 0, 0])
    >>> recovered, remainder = signal.deconvolve(recorded, impulse_response)
    >>> recovered
    array([ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.])

    """
    num = np.atleast_1d(signal)
    den = np.atleast_1d(divisor)
    if num.ndim > 1:
        raise ValueError('signal must be 1-D.')
    if den.ndim > 1:
        raise ValueError('divisor must be 1-D.')
    N = len(num)
    D = len(den)
    if D > N:
        quot = []
        rem = num
    else:
        input = np.zeros(N - D + 1, float)
        input[0] = 1
        quot = lfilter(num, den, input)
        rem = num - convolve(den, quot, mode='full')
    return (quot, rem)