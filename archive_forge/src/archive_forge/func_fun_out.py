from warnings import warn
import numpy as np
from ._optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr
from ._lbfgsb_py import _minimize_lbfgsb
from ._tnc import _minimize_tnc
from ._cobyla_py import _minimize_cobyla
from ._slsqp_py import _minimize_slsqp
from ._constraints import (old_bound_to_new, new_bounds_to_old,
from ._differentiable_functions import FD_METHODS
def fun_out(x_in, *args, **kwargs):
    x_out = np.zeros_like(i_fixed, dtype=x_in.dtype)
    x_out[i_fixed] = x_fixed
    x_out[~i_fixed] = x_in
    y_out = fun_in(x_out, *args, **kwargs)
    y_out = np.array(y_out)
    if min_dim == 1:
        y_out = np.atleast_1d(y_out)
    elif min_dim == 2:
        y_out = np.atleast_2d(y_out)
    if remove == 1:
        y_out = y_out[..., ~i_fixed]
    elif remove == 2:
        y_out = y_out[~i_fixed, ~i_fixed]
    return y_out