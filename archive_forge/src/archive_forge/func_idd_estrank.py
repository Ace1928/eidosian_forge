import scipy.linalg._interpolative as _id
import numpy as np
def idd_estrank(eps, A):
    """
    Estimate rank of a real matrix to a specified relative precision using
    random sampling.

    The output rank is typically about 8 higher than the actual rank.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank estimate.
    :rtype: int
    """
    A = np.asfortranarray(A)
    m, n = A.shape
    n2, w = idd_frmi(m)
    ra = np.empty(n * n2 + (n + 1) * (n2 + 1), order='F')
    k, ra = _id.idd_estrank(eps, A, w, ra)
    return k