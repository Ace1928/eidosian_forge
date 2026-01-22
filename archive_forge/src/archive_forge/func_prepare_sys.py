from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h):
    """Create the function and the Jacobian for the collocation system."""
    x_middle = x[:-1] + 0.5 * h
    i_jac, j_jac = compute_jac_indices(n, m, k)

    def col_fun(y, p):
        return collocation_fun(fun, y, p, x, h)

    def sys_jac(y, p, y_middle, f, f_middle, bc0):
        if fun_jac is None:
            df_dy, df_dp = estimate_fun_jac(fun, x, y, p, f)
            df_dy_middle, df_dp_middle = estimate_fun_jac(fun, x_middle, y_middle, p, f_middle)
        else:
            df_dy, df_dp = fun_jac(x, y, p)
            df_dy_middle, df_dp_middle = fun_jac(x_middle, y_middle, p)
        if bc_jac is None:
            dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(bc, y[:, 0], y[:, -1], p, bc0)
        else:
            dbc_dya, dbc_dyb, dbc_dp = bc_jac(y[:, 0], y[:, -1], p)
        return construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    return (col_fun, sys_jac)