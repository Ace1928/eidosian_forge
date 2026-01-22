import numpy as np
from ._ufuncs import _ellip_harm
from ._ellip_harm_2 import _ellipsoid, _ellipsoid_norm
def ellip_normal(h2, k2, n, p):
    """
    Ellipsoidal harmonic normalization constants gamma^p_n

    The normalization constant is defined as

    .. math::

       \\gamma^p_n=8\\int_{0}^{h}dx\\int_{h}^{k}dy
       \\frac{(y^2-x^2)(E^p_n(y)E^p_n(x))^2}{\\sqrt((k^2-y^2)(y^2-h^2)(h^2-x^2)(k^2-x^2)}

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree.
    p : int
        Order, can range between [1,2n+1].

    Returns
    -------
    gamma : float
        The normalization constant :math:`\\gamma^p_n`

    See Also
    --------
    ellip_harm, ellip_harm_2

    Notes
    -----
    .. versionadded:: 0.15.0

    Examples
    --------
    >>> from scipy.special import ellip_normal
    >>> w = ellip_normal(5,8,3,7)
    >>> w
    1723.38796997

    """
    with np.errstate(all='ignore'):
        return _ellip_normal_vec(h2, k2, n, p)