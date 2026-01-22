import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def sos2zpk(sos):
    """
    Return zeros, poles, and gain of a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    The number of zeros and poles returned will be ``n_sections * 2``
    even if some of these are (effectively) zero.

    See Also
    --------
    scipy.signal.sos2zpk

    """
    n_sections = sos.shape[0]
    z = cupy.zeros(n_sections * 2, cupy.complex128)
    p = cupy.zeros(n_sections * 2, cupy.complex128)
    k = 1.0
    for section in range(n_sections):
        zpk = tf2zpk(sos[section, :3], sos[section, 3:])
        z[2 * section:2 * section + len(zpk[0])] = zpk[0]
        p[2 * section:2 * section + len(zpk[1])] = zpk[1]
        k *= zpk[2]
    return (z, p, k)