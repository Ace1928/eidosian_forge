from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def bc_jac_wrapped(ya, yb, p):
    dbc_dya, dbc_dyb, dbc_dp = bc_jac(ya, yb, p)
    return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype), np.asarray(dbc_dp, dtype))