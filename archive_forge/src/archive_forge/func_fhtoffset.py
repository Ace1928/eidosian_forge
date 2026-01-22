import numpy as np
from warnings import warn
from ._basic import rfft, irfft
from ..special import loggamma, poch
from scipy._lib._array_api import array_namespace, copy
def fhtoffset(dln, mu, initial=0.0, bias=0.0):
    """Return optimal offset for a fast Hankel transform.

    Returns an offset close to `initial` that fulfils the low-ringing
    condition of [1]_ for the fast Hankel transform `fht` with logarithmic
    spacing `dln`, order `mu` and bias `bias`.

    Parameters
    ----------
    dln : float
        Uniform logarithmic spacing of the transform.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    initial : float, optional
        Initial value for the offset. Returns the closest value that fulfils
        the low-ringing condition.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    offset : float
        Optimal offset of the uniform logarithmic spacing of the transform that
        fulfils a low-ringing condition.

    Examples
    --------
    >>> from scipy.fft import fhtoffset
    >>> dln = 0.1
    >>> mu = 2.0
    >>> initial = 0.5
    >>> bias = 0.0
    >>> offset = fhtoffset(dln, mu, initial, bias)
    >>> offset
    0.5454581477676637

    See Also
    --------
    fht : Definition of the fast Hankel transform.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    """
    lnkr, q = (initial, bias)
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = np.pi / (2 * dln)
    zp = loggamma(xp + 1j * y)
    zm = loggamma(xm + 1j * y)
    arg = (LN_2 - lnkr) / dln + (zp.imag + zm.imag) / np.pi
    return lnkr + (arg - np.round(arg)) * dln