import scipy.linalg._interpolative as _id
import numpy as np
def iddr_id(A, k):
    """
    Compute ID of a real matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    A = _asfortranarray_copy(A)
    idx, rnorms = _id.iddr_id(A, k)
    n = A.shape[1]
    proj = A.T.ravel()[:k * (n - k)].reshape((k, n - k), order='F')
    return (idx, proj)