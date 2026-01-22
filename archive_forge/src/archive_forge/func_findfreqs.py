import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
def findfreqs(num, den, N, kind='ba'):
    """
    Find array of frequencies for computing the response of an analog filter.

    Parameters
    ----------
    num, den : array_like, 1-D
        The polynomial coefficients of the numerator and denominator of the
        transfer function of the filter or LTI system, where the coefficients
        are ordered from highest to lowest degree. Or, the roots  of the
        transfer function numerator and denominator (i.e., zeroes and poles).
    N : int
        The length of the array to be computed.
    kind : str {'ba', 'zp'}, optional
        Specifies whether the numerator and denominator are specified by their
        polynomial coefficients ('ba'), or their roots ('zp').

    Returns
    -------
    w : (N,) ndarray
        A 1-D array of frequencies, logarithmically spaced.

    Warning
    -------
    This function may synchronize the device.

    See Also
    --------
    scipy.signal.find_freqs

    Examples
    --------
    Find a set of nine frequencies that span the "interesting part" of the
    frequency response for the filter with the transfer function

        H(s) = s / (s^2 + 8s + 25)

    >>> from scipy import signal
    >>> signal.findfreqs([1, 0], [1, 8, 25], N=9)
    array([  1.00000000e-02,   3.16227766e-02,   1.00000000e-01,
             3.16227766e-01,   1.00000000e+00,   3.16227766e+00,
             1.00000000e+01,   3.16227766e+01,   1.00000000e+02])
    """
    if kind == 'ba':
        ep = cupy.atleast_1d(roots(den)) + 0j
        tz = cupy.atleast_1d(roots(num)) + 0j
    elif kind == 'zp':
        ep = cupy.atleast_1d(den) + 0j
        tz = cupy.atleast_1d(num) + 0j
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")
    if len(ep) == 0:
        ep = cupy.atleast_1d(-1000) + 0j
    ez = cupy.r_[cupy.compress(ep.imag >= 0, ep, axis=-1), cupy.compress((abs(tz) < 100000.0) & (tz.imag >= 0), tz, axis=-1)]
    integ = cupy.abs(ez) < 1e-10
    hfreq = cupy.around(cupy.log10(cupy.max(3 * cupy.abs(ez.real + integ) + 1.5 * ez.imag)) + 0.5)
    lfreq = cupy.around(cupy.log10(0.1 * cupy.min(cupy.abs((ez + integ).real) + 2 * ez.imag)) - 0.5)
    w = cupy.logspace(lfreq, hfreq, N)
    return w