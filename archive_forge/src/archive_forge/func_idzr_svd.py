import scipy.linalg._interpolative as _id
import numpy as np
def idzr_svd(A, k):
    """
    Compute SVD of a complex matrix to a specified rank.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    A = np.asfortranarray(A)
    U, V, S, ier = _id.idzr_svd(A, k)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)