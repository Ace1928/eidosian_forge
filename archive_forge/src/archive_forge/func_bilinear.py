import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def bilinear(b, a, fs=1.0):
    """
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    b : array_like
        Numerator of the analog filter transfer function.
    a : array_like
        Denominator of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    b : ndarray
        Numerator of the transformed digital filter transfer function.
    a : ndarray
        Denominator of the transformed digital filter transfer function.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, lp2bs
    bilinear_zpk
    scipy.signal.bilinear

    """
    fs = float(fs)
    a, b = map(cupy.atleast_1d, (a, b))
    D = a.shape[0] - 1
    N = b.shape[0] - 1
    M = max(N, D)
    Np, Dp = (M, M)
    bprime = cupy.empty(Np + 1, float)
    aprime = cupy.empty(Dp + 1, float)
    for j in range(Dp + 1):
        val = 0.0
        for i in range(N + 1):
            bNi = b[N - i] * (2 * fs) ** i
            for k in range(i + 1):
                for s in range(M - i + 1):
                    if k + s == j:
                        val += comb(i, k) * comb(M - i, s) * bNi * (-1) ** k
        bprime[j] = cupy.real(val)
    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            aDi = a[D - i] * (2 * fs) ** i
            for k in range(i + 1):
                for s in range(M - i + 1):
                    if k + s == j:
                        val += comb(i, k) * comb(M - i, s) * aDi * (-1) ** k
        aprime[j] = cupy.real(val)
    return normalize(bprime, aprime)