import scipy.linalg._interpolative as _id
import numpy as np
def idzp_id(eps, A):
    """
    Compute ID of a complex matrix to a specified relative precision.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    A = _asfortranarray_copy(A)
    k, idx, rnorms = _id.idzp_id(eps, A)
    n = A.shape[1]
    proj = A.T.ravel()[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)