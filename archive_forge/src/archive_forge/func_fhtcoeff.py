import math
from warnings import warn
import cupy
from cupyx.scipy.fft import _fft
from cupyx.scipy.special import loggamma, poch
def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0):
    """Compute the coefficient array for a fast Hankel transform.
    """
    lnkr, q = (offset, bias)
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = cupy.linspace(0, math.pi * (n // 2) / (n * dln), n // 2 + 1)
    u = cupy.empty(n // 2 + 1, dtype=complex)
    v = cupy.empty(n // 2 + 1, dtype=complex)
    u.imag[:] = y
    u.real[:] = xm
    loggamma(u, out=v)
    u.real[:] = xp
    loggamma(u, out=u)
    y *= 2 * (LN_2 - lnkr)
    u.real -= v.real
    u.real += LN_2 * q
    u.imag += v.imag
    u.imag += y
    cupy.exp(u, out=u)
    u.imag[-1] = 0
    if not cupy.isfinite(u[0]):
        u[0] = 2 ** q * poch(xm, xp - xm)
    return u