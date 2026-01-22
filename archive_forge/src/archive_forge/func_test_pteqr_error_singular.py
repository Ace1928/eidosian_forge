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
@pytest.mark.parametrize('dtype,realtype', zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize('compute_z', range(3))
def test_pteqr_error_singular(dtype, realtype, compute_z):
    seed(42)
    pteqr = get_lapack_funcs('pteqr', dtype=dtype)
    n = 10
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)
    d[0] = 0
    e[0] = 0
    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d, e, z=z, compute_z=compute_z)
    assert info > 0