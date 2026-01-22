import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    """
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p : ndarray
        Poles of the transformed band-stop filter transfer function.
    k : float
        System gain of the transformed band-stop filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, bilinear
    lp2bs
    scipy.signal.lp2bs_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \\rightarrow \\frac{s \\cdot \\mathrm{BW}}{s^2 + {\\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    """
    z = cupy.atleast_1d(z)
    p = cupy.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)
    degree = _relative_degree(z, p)
    z_hp = bw / 2 / z
    p_hp = bw / 2 / p
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)
    z_bs = cupy.concatenate((z_hp + cupy.sqrt(z_hp ** 2 - wo ** 2), z_hp - cupy.sqrt(z_hp ** 2 - wo ** 2)))
    p_bs = cupy.concatenate((p_hp + cupy.sqrt(p_hp ** 2 - wo ** 2), p_hp - cupy.sqrt(p_hp ** 2 - wo ** 2)))
    z_bs = cupy.append(z_bs, cupy.full(degree, +1j * wo))
    z_bs = cupy.append(z_bs, cupy.full(degree, -1j * wo))
    k_bs = k * cupy.real(cupy.prod(-z) / cupy.prod(-p))
    return (z_bs, p_bs, k_bs)