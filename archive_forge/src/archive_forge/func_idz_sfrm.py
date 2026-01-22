import scipy.linalg._interpolative as _id
import numpy as np
def idz_sfrm(l, n, w, x):
    """
    Transform complex vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idz_frm`, this routine works best when the length of
    the transformed vector is known a priori.

    :param l:
        Length of transformed vector, satisfying `l <= n`.
    :type l: int
    :param n:
        Greatest power-of-two integer satisfying `n <= x.size` as obtained from
        :func:`idz_sfrmi`.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idd_sfrmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idz_sfrm(l, n, w, x)