from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def fun_jac_p(x, y, p):
    df_dy, df_dp = fun_jac(x, y, p)
    return (np.asarray(df_dy, dtype), np.asarray(df_dp, dtype))