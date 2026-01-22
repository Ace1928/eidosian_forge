import numpy as np
from scipy.special import erf
def tricube(h, Xi, x):
    """
    Tricube Kernel for continuous variables
    Parameters
    ----------
    h : 1-D ndarray, shape (K,)
        The bandwidths used to estimate the value of the kernel function.
    Xi : 1-D ndarray, shape (K,)
        The value of the training set.
    x : 1-D ndarray, shape (K,)
        The value at which the kernel density is being estimated.

    Returns
    -------
    kernel_value : ndarray, shape (nobs, K)
        The value of the kernel function at each training point for each var.
    """
    u = (Xi - x) / h
    u[np.abs(u) > 1] = 0
    return 70.0 / 81 * (1 - np.abs(u) ** 3) ** 3