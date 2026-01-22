import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def cheb2ap(N, rs):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rs` decibels of ripple in the stopband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first reaches ``-rs``.

    See Also
    --------
    cheby2 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError('Filter order must be a nonnegative integer')
    elif N == 0:
        return (cupy.array([]), cupy.array([]), 1)
    de = 1.0 / cupy.sqrt(10 ** (0.1 * rs) - 1)
    mu = cupy.arcsinh(1.0 / de) / N
    if N % 2:
        m = cupy.concatenate((cupy.arange(-N + 1, 0, 2), cupy.arange(2, N, 2)))
    else:
        m = cupy.arange(-N + 1, N, 2)
    z = -cupy.conjugate(1j / cupy.sin(m * pi / (2.0 * N)))
    p = -cupy.exp(1j * pi * cupy.arange(-N + 1, N, 2) / (2 * N))
    p = cupy.sinh(mu) * p.real + 1j * cupy.cosh(mu) * p.imag
    p = 1.0 / p
    k = (cupy.prod(-p, axis=0) / cupy.prod(-z, axis=0)).real
    return (z, p, k)