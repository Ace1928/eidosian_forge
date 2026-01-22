from warnings import warn
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult
def fun_jac_wrapped(x, y, p):
    df_dy, df_dp = fun_jac_p(x, y, p)
    if x[0] == a:
        df_dy[:, :, 0] = np.dot(D, df_dy[:, :, 0])
        df_dy[:, :, 1:] += Sr / (x[1:] - a)
    else:
        df_dy += Sr / (x - a)
    return (df_dy, df_dp)