import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def lp2hp(b, a, wo=1.0):
    """
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed high-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed high-pass
        filter.

    See Also
    --------
    lp2lp, lp2bp, lp2bs, bilinear
    lp2hp_zpk
    scipy.signal.lp2hp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \\rightarrow \\frac{\\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.
    """
    a, b = map(cupy.atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    if wo != 1:
        pwo = wo ** cupy.arange(max(d, n))
    else:
        pwo = cupy.ones(max(d, n), b.dtype)
    if d >= n:
        outa = a[::-1] * pwo
        outb = cupy.resize(b, (d,))
        outb[n:] = 0.0
        outb[:n] = b[::-1] * pwo[:n]
    else:
        outb = b[::-1] * pwo
        outa = cupy.resize(a, (n,))
        outa[d:] = 0.0
        outa[:d] = a[::-1] * pwo[:d]
    return normalize(outb, outa)