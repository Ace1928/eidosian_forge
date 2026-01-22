import scipy.linalg._interpolative as _id
import numpy as np
def idd_id2svd(B, idx, proj):
    """
    Convert real ID to SVD.

    :param B:
        Skeleton matrix.
    :type B: :class:`numpy.ndarray`
    :param idx:
        Column index array.
    :type idx: :class:`numpy.ndarray`
    :param proj:
        Interpolation coefficients.
    :type proj: :class:`numpy.ndarray`

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
    B = np.asfortranarray(B)
    U, V, S, ier = _id.idd_id2svd(B, idx, proj)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)