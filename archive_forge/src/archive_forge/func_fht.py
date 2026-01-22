import math
from warnings import warn
import cupy
from cupyx.scipy.fft import _fft
from cupyx.scipy.special import loggamma, poch
@_fft._implements(_scipy_fft.fht)
def fht(a, dln, mu, offset=0.0, bias=0.0):
    """Compute the fast Hankel transform.

    Computes the discrete Hankel transform of a logarithmically spaced periodic
    sequence using the FFTLog algorithm [1]_, [2]_.

    Parameters
    ----------
    a : cupy.ndarray (..., n)
        Real periodic input array, uniformly logarithmically spaced.  For
        multidimensional input, the transform is performed over the last axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    offset : float, optional
        Offset of the uniform logarithmic spacing of the output array.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    A : cupy.ndarray (..., n)
        The transformed output array, which is real, periodic, uniformly
        logarithmically spaced, and of the same shape as the input array.

    See Also
    --------
    :func:`scipy.special.fht`
    :func:`scipy.special.fhtoffset` : Return an optimal offset for `fht`.

    References
    ----------
    .. [1] Talman J. D., 1978, J. Comp. Phys., 29, 35
    .. [2] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    """
    n = a.shape[-1]
    if bias != 0:
        j_c = (n - 1) / 2
        j = cupy.arange(n)
        a = a * cupy.exp(-bias * (j - j_c) * dln)
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)
    A = _fhtq(a, u)
    if bias != 0:
        A *= cupy.exp(-bias * ((j - j_c) * dln + offset))
    return A