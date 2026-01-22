from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def intersect_trust_region(x, s, Delta):
    """Find the intersection of a line with the boundary of a trust region.

    This function solves the quadratic equation with respect to t
    ||(x + s*t)||**2 = Delta**2.

    Returns
    -------
    t_neg, t_pos : tuple of float
        Negative and positive roots.

    Raises
    ------
    ValueError
        If `s` is zero or `x` is not within the trust region.
    """
    a = np.dot(s, s)
    if a == 0:
        raise ValueError('`s` is zero.')
    b = np.dot(x, s)
    c = np.dot(x, x) - Delta ** 2
    if c > 0:
        raise ValueError('`x` is not within the trust region.')
    d = np.sqrt(b * b - a * c)
    q = -(b + copysign(d, b))
    t1 = q / a
    t2 = c / q
    if t1 < t2:
        return (t1, t2)
    else:
        return (t2, t1)