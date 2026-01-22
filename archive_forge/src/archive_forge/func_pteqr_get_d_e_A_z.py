import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def pteqr_get_d_e_A_z(dtype, realtype, n, compute_z):
    if compute_z == 1:
        A_eig = generate_random_dtype_array((n, n), dtype)
        A_eig = A_eig + np.diag(np.zeros(n) + 4 * n)
        A_eig = (A_eig + A_eig.conj().T) / 2
        vr = eigh(A_eig)[1]
        d = generate_random_dtype_array((n,), realtype) + 4
        e = generate_random_dtype_array((n - 1,), realtype)
        tri = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        A = vr @ tri @ vr.conj().T
        z = vr
    else:
        d = generate_random_dtype_array((n,), realtype)
        e = generate_random_dtype_array((n - 1,), realtype)
        d = d + 4
        A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        z = np.diag(d) + np.diag(e, -1) + np.diag(e, 1)
    return (d, e, A, z)