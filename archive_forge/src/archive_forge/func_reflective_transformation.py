from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def reflective_transformation(y, lb, ub):
    """Compute reflective transformation and its gradient."""
    if in_bounds(y, lb, ub):
        return (y, np.ones_like(y))
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)
    x = y.copy()
    g_negative = np.zeros_like(y, dtype=bool)
    mask = lb_finite & ~ub_finite
    x[mask] = np.maximum(y[mask], 2 * lb[mask] - y[mask])
    g_negative[mask] = y[mask] < lb[mask]
    mask = ~lb_finite & ub_finite
    x[mask] = np.minimum(y[mask], 2 * ub[mask] - y[mask])
    g_negative[mask] = y[mask] > ub[mask]
    mask = lb_finite & ub_finite
    d = ub - lb
    t = np.remainder(y[mask] - lb[mask], 2 * d[mask])
    x[mask] = lb[mask] + np.minimum(t, 2 * d[mask] - t)
    g_negative[mask] = t > d[mask]
    g = np.ones_like(y)
    g[g_negative] = -1
    return (x, g)