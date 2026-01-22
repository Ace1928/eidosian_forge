import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def band_stop_obj(wp, ind, passb, stopb, gpass, gstop, type):
    """
    Band Stop Objective Function for order minimization.

    Returns the non-integer order for an analog band stop filter.

    Parameters
    ----------
    wp : scalar
        Edge of passband `passb`.
    ind : int, {0, 1}
        Index specifying which `passb` edge to vary (0 or 1).
    passb : ndarray
        Two element sequence of fixed passband edges.
    stopb : ndarray
        Two element sequence of fixed stopband edges.
    gstop : float
        Amount of attenuation in stopband in dB.
    gpass : float
        Amount of ripple in the passband in dB.
    type : {'butter', 'cheby', 'ellip'}
        Type of filter.

    Returns
    -------
    n : scalar
        Filter order (possibly non-integer).

    See Also
    --------
    scipy.signal.band_stop_obj

    """
    _validate_gpass_gstop(gpass, gstop)
    passbC = passb.copy()
    passbC[ind] = wp
    nat = stopb * (passbC[0] - passbC[1]) / (stopb ** 2 - passbC[0] * passbC[1])
    nat = min(cupy.abs(nat))
    if type == 'butter':
        GSTOP = 10 ** (0.1 * cupy.abs(gstop))
        GPASS = 10 ** (0.1 * cupy.abs(gpass))
        n = cupy.log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * cupy.log10(nat))
    elif type == 'cheby':
        GSTOP = 10 ** (0.1 * cupy.abs(gstop))
        GPASS = 10 ** (0.1 * cupy.abs(gpass))
        n = cupy.arccosh(cupy.sqrt((GSTOP - 1.0) / (GPASS - 1.0))) / cupy.arccosh(nat)
    elif type == 'ellip':
        GSTOP = 10 ** (0.1 * gstop)
        GPASS = 10 ** (0.1 * gpass)
        arg1 = cupy.sqrt((GPASS - 1.0) / (GSTOP - 1.0))
        arg0 = 1.0 / nat
        d0 = special.ellipk(cupy.array([arg0 ** 2, 1 - arg0 ** 2]))
        d1 = special.ellipk(cupy.array([arg1 ** 2, 1 - arg1 ** 2]))
        n = d0[0] * d1[1] / (d0[1] * d1[0])
    else:
        raise ValueError('Incorrect type: %s' % type)
    return n