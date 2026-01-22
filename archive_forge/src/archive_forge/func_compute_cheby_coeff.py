import numpy as np
from scipy import sparse
from pygsp import utils
@utils.filterbank_handler
def compute_cheby_coeff(f, m=30, N=None, *args, **kwargs):
    """
    Compute Chebyshev coefficients for a Filterbank.

    Parameters
    ----------
    f : Filter
        Filterbank with at least 1 filter
    m : int
        Maximum order of Chebyshev coeff to compute
        (default = 30)
    N : int
        Grid order used to compute quadrature
        (default = m + 1)
    i : int
        Index of the Filterbank element to compute
        (default = 0)

    Returns
    -------
    c : ndarray
        Matrix of Chebyshev coefficients

    """
    G = f.G
    i = kwargs.pop('i', 0)
    if not N:
        N = m + 1
    a_arange = [0, G.lmax]
    a1 = (a_arange[1] - a_arange[0]) / 2
    a2 = (a_arange[1] + a_arange[0]) / 2
    c = np.zeros(m + 1)
    tmpN = np.arange(N)
    num = np.cos(np.pi * (tmpN + 0.5) / N)
    for o in range(m + 1):
        c[o] = 2.0 / N * np.dot(f._kernels[i](a1 * num + a2), np.cos(np.pi * o * (tmpN + 0.5) / N))
    return c