import scipy.linalg._interpolative as _id
import numpy as np
def idd_frm(n, w, x):
    """
    Transform real vector via a composition of Rokhlin's random transform,
    random subselection, and an FFT.

    In contrast to :func:`idd_sfrm`, this routine works best when the length of
    the transformed vector is the power-of-two integer output by
    :func:`idd_frmi`, or when the length is not specified but instead
    determined a posteriori from the output. The returned transformed vector is
    randomly permuted.

    :param n:
        Greatest power-of-two integer satisfying `n <= x.size` as obtained from
        :func:`idd_frmi`; `n` is also the length of the output vector.
    :type n: int
    :param w:
        Initialization array constructed by :func:`idd_frmi`.
    :type w: :class:`numpy.ndarray`
    :param x:
        Vector to be transformed.
    :type x: :class:`numpy.ndarray`

    :return:
        Transformed vector.
    :rtype: :class:`numpy.ndarray`
    """
    return _id.idd_frm(n, w, x)