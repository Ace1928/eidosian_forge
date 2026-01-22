from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def solve_trust_region_2d(B, g, Delta):
    """Solve a general trust-region problem in 2 dimensions.

    The problem is reformulated as a 4th order algebraic equation,
    the solution of which is found by numpy.roots.

    Parameters
    ----------
    B : ndarray, shape (2, 2)
        Symmetric matrix, defines a quadratic term of the function.
    g : ndarray, shape (2,)
        Defines a linear term of the function.
    Delta : float
        Radius of a trust region.

    Returns
    -------
    p : ndarray, shape (2,)
        Found solution.
    newton_step : bool
        Whether the returned solution is the Newton step which lies within
        the trust region.
    """
    try:
        R, lower = cho_factor(B)
        p = -cho_solve((R, lower), g)
        if np.dot(p, p) <= Delta ** 2:
            return (p, True)
    except LinAlgError:
        pass
    a = B[0, 0] * Delta ** 2
    b = B[0, 1] * Delta ** 2
    c = B[1, 1] * Delta ** 2
    d = g[0] * Delta
    f = g[1] * Delta
    coeffs = np.array([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.roots(coeffs)
    t = np.real(t[np.isreal(t)])
    p = Delta * np.vstack((2 * t / (1 + t ** 2), (1 - t ** 2) / (1 + t ** 2)))
    value = 0.5 * np.sum(p * B.dot(p), axis=0) + np.dot(g, p)
    i = np.argmin(value)
    p = p[:, i]
    return (p, False)