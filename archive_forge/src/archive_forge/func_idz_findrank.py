import scipy.linalg._interpolative as _id
import numpy as np
def idz_findrank(eps, m, n, matveca):
    """
    Estimate rank of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function

    :return:
        Rank estimate.
    :rtype: int
    """
    k, ra, ier = _id.idz_findrank(eps, m, n, matveca)
    if ier:
        raise _RETCODE_ERROR
    return k