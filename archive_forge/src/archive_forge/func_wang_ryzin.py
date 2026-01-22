import numpy as np
from scipy.special import erf
def wang_ryzin(h, Xi, x):
    """
    The Wang-Ryzin kernel, used for ordered discrete random variables.

    Parameters
    ----------
    h : scalar or 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : ndarray of ints, shape (nobs, K)
        The value of the training set.
    x : scalar or 1-D ndarray of shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.

    Notes
    -----
    See p. 19 in [1]_ for details.  The value of the kernel L if
    :math:`X_{i}=x` is :math:`1-\\lambda`, otherwise it is
    :math:`\\frac{1-\\lambda}{2}\\lambda^{|X_{i}-x|}`, where :math:`\\lambda` is
    the bandwidth.

    References
    ----------
    .. [*] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
           http://dx.doi.org/10.1561/0800000009
    .. [*] M.-C. Wang and J. van Ryzin, "A class of smooth estimators for
           discrete distributions", Biometrika, vol. 68, pp. 301-309, 1981.
    """
    Xi = Xi.reshape(Xi.size)
    kernel_value = 0.5 * (1 - h) * h ** abs(Xi - x)
    idx = Xi == x
    kernel_value[idx] = (idx * (1 - h))[idx]
    return kernel_value