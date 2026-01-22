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
def test_geequ():
    desired_real = np.array([[0.625, 1.0, 0.0393, -0.4269], [1.0, -0.5619, -1.0, -1.0], [0.5874, -1.0, -0.0596, -0.5341], [-1.0, -0.5946, -0.0294, 0.9957]])
    desired_cplx = np.array([[-0.2816 + 0.5359 * 1j, 0.0812 + 0.9188 * 1j, -0.7439 - 0.2561 * 1j], [-0.3562 - 0.2954 * 1j, 0.9566 - 0.0434 * 1j, -0.0174 + 0.1555 * 1j], [0.8607 + 0.1393 * 1j, -0.2759 + 0.7241 * 1j, -0.1642 - 0.1365 * 1j]])
    for ind, dtype in enumerate(DTYPES):
        if ind < 2:
            A = np.array([[18000000000.0, 28800000000.0, 2.05, -8900000000.0], [5.25, -2.95, -9.5e-09, -3.8], [1.58, -2.69, -2.9e-10, -1.04], [-1.11, -0.66, -5.9e-11, 0.8]])
            A = A.astype(dtype)
        else:
            A = np.array([[-1.34, 2800000000.0, -6.39], [-1.7, 33100000000.0, -0.15], [2.41e-10, -0.56, -8.3e-11]], dtype=dtype)
            A += np.array([[2.55, 31700000000.0, -2.2], [-1.41, -1500000000.0, 1.34], [3.9e-11, 1.47, -6.9e-11]]) * 1j
            A = A.astype(dtype)
        geequ = get_lapack_funcs('geequ', dtype=dtype)
        r, c, rowcnd, colcnd, amax, info = geequ(A)
        if ind < 2:
            assert_allclose(desired_real.astype(dtype), r[:, None] * A * c, rtol=0, atol=0.0001)
        else:
            assert_allclose(desired_cplx.astype(dtype), r[:, None] * A * c, rtol=0, atol=0.0001)