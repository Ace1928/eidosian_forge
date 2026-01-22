import numpy as np
from numpy.linalg import lstsq, norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy.optimize import OptimizeResult
from .common import (
def find_intersection(x, tr_bounds, lb, ub):
    """Find intersection of trust-region bounds and initial bounds.

    Returns
    -------
    lb_total, ub_total : ndarray with shape of x
        Lower and upper bounds of the intersection region.
    orig_l, orig_u : ndarray of bool with shape of x
        True means that an original bound is taken as a corresponding bound
        in the intersection region.
    tr_l, tr_u : ndarray of bool with shape of x
        True means that a trust-region bound is taken as a corresponding bound
        in the intersection region.
    """
    lb_centered = lb - x
    ub_centered = ub - x
    lb_total = np.maximum(lb_centered, -tr_bounds)
    ub_total = np.minimum(ub_centered, tr_bounds)
    orig_l = np.equal(lb_total, lb_centered)
    orig_u = np.equal(ub_total, ub_centered)
    tr_l = np.equal(lb_total, -tr_bounds)
    tr_u = np.equal(ub_total, tr_bounds)
    return (lb_total, ub_total, orig_l, orig_u, tr_l, tr_u)