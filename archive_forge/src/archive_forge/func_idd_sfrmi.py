import scipy.linalg._interpolative as _id
import numpy as np
def idd_sfrmi(l, m):
    """
    Initialize data for :func:`idd_sfrm`.

    :param l:
        Length of output transformed vector.
    :type l: int
    :param m:
        Length of the vector to be transformed.
    :type m: int

    :return:
        Greatest power-of-two integer `n` satisfying `n <= m`.
    :rtype: int
    :return:
        Initialization array to be used by :func:`idd_sfrm`.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idd_sfrmi(l, m)