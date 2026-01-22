import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, concatenate, finfo,
from ._optimize import (OptimizeResult, _check_unknown_options,
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import atleast_nd, array_namespace
from numpy import exp, inf  # noqa: F401
def cjac_factory(fun):

    def cjac(x, *args):
        x = _check_clip_x(x, new_bounds)
        if jac in ['2-point', '3-point', 'cs']:
            return approx_derivative(fun, x, method=jac, args=args, rel_step=finite_diff_rel_step, bounds=new_bounds)
        else:
            return approx_derivative(fun, x, method='2-point', abs_step=epsilon, args=args, bounds=new_bounds)
    return cjac