import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult
from .common import (
def trf(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose):
    if np.all(lb == -np.inf) and np.all(ub == np.inf):
        return trf_no_bounds(fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose)
    else:
        return trf_bounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose)