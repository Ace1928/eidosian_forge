from warnings import warn
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import _minpack, OptimizeResult
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.optimize._minimize import Bounds
from .trf import trf
from .dogbox import dogbox
from .common import EPS, in_bounds, make_strictly_feasible
def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step):
    n = x0.size
    if diff_step is None:
        epsfcn = EPS
    else:
        epsfcn = diff_step ** 2
    if isinstance(x_scale, str) and x_scale == 'jac':
        diag = None
    else:
        diag = 1 / x_scale
    full_output = True
    col_deriv = False
    factor = 100.0
    if jac is None:
        if max_nfev is None:
            max_nfev = 100 * n * (n + 1)
        x, info, status = _minpack._lmdif(fun, x0, (), full_output, ftol, xtol, gtol, max_nfev, epsfcn, factor, diag)
    else:
        if max_nfev is None:
            max_nfev = 100 * n
        x, info, status = _minpack._lmder(fun, jac, x0, (), full_output, col_deriv, ftol, xtol, gtol, max_nfev, factor, diag)
    f = info['fvec']
    if callable(jac):
        J = jac(x)
    else:
        J = np.atleast_2d(approx_derivative(fun, x))
    cost = 0.5 * np.dot(f, f)
    g = J.T.dot(f)
    g_norm = norm(g, ord=np.inf)
    nfev = info['nfev']
    njev = info.get('njev', None)
    status = FROM_MINPACK_TO_COMMON[status]
    active_mask = np.zeros_like(x0, dtype=int)
    return OptimizeResult(x=x, cost=cost, fun=f, jac=J, grad=g, optimality=g_norm, active_mask=active_mask, nfev=nfev, njev=njev, status=status)