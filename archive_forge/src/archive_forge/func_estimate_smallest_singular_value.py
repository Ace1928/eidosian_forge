import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def estimate_smallest_singular_value(U):
    """Given upper triangular matrix ``U`` estimate the smallest singular
    value and the correspondent right singular vector in O(n**2) operations.

    Parameters
    ----------
    U : ndarray
        Square upper triangular matrix.

    Returns
    -------
    s_min : float
        Estimated smallest singular value of the provided matrix.
    z_min : ndarray
        Estimatied right singular vector.

    Notes
    -----
    The procedure is based on [1]_ and is done in two steps. First, it finds
    a vector ``e`` with components selected from {+1, -1} such that the
    solution ``w`` from the system ``U.T w = e`` is as large as possible.
    Next it estimate ``U v = w``. The smallest singular value is close
    to ``norm(w)/norm(v)`` and the right singular vector is close
    to ``v/norm(v)``.

    The estimation will be better more ill-conditioned is the matrix.

    References
    ----------
    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
           An estimate for the condition number of a matrix.  1979.
           SIAM Journal on Numerical Analysis, 16(2), 368-375.
    """
    U = np.atleast_2d(U)
    m, n = U.shape
    if m != n:
        raise ValueError('A square triangular matrix should be provided.')
    p = np.zeros(n)
    w = np.empty(n)
    for k in range(n):
        wp = (1 - p[k]) / U.T[k, k]
        wm = (-1 - p[k]) / U.T[k, k]
        pp = p[k + 1:] + U.T[k + 1:, k] * wp
        pm = p[k + 1:] + U.T[k + 1:, k] * wm
        if abs(wp) + norm(pp, 1) >= abs(wm) + norm(pm, 1):
            w[k] = wp
            p[k + 1:] = pp
        else:
            w[k] = wm
            p[k + 1:] = pm
    v = solve_triangular(U, w)
    v_norm = norm(v)
    w_norm = norm(w)
    s_min = w_norm / v_norm
    z_min = v / v_norm
    return (s_min, z_min)