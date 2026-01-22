import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm
def bg_update_dense(plu, perm_r, v, j):
    LU, p = plu
    vperm = v[perm_r]
    u = dtrsm(1, LU, vperm, lower=1, diag=1)
    LU[:j + 1, j] = u[:j + 1]
    l = u[j + 1:]
    piv = LU[j, j]
    LU[j + 1:, j] += l / piv
    return (LU, p)