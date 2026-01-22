import scipy.linalg._interpolative as _id
import numpy as np
def iddr_aid(A, k):
    """
    Compute ID of a real matrix to a specified rank using random sampling.

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
    A = np.asfortranarray(A)
    m, n = A.shape
    w = iddr_aidi(m, n, k)
    idx, proj = _id.iddr_aid(A, k, w)
    if k == n:
        proj = np.empty((k, n - k), dtype='float64', order='F')
    else:
        proj = proj.reshape((k, n - k), order='F')
    return (idx, proj)