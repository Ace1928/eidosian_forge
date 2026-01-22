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
@pytest.mark.parametrize('a, b, c, d, e, f, rans, lans', [(np.array([[4.0, 1.0, 1.0, 2.0], [0.0, 3.0, 4.0, 1.0], [0.0, 1.0, 3.0, 1.0], [0.0, 0.0, 0.0, 6.0]]), np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 3.0, 4.0, 1.0], [0.0, 1.0, 3.0, 1.0], [0.0, 0.0, 0.0, 4.0]]), np.array([[-4.0, 7.0, 1.0, 12.0], [-9.0, 2.0, -2.0, -2.0], [-4.0, 2.0, -2.0, 8.0], [-7.0, 7.0, -6.0, 19.0]]), np.array([[2.0, 1.0, 1.0, 3.0], [0.0, 1.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 2.0]]), np.array([[1.0, 1.0, 1.0, 2.0], [0.0, 1.0, 4.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]), np.array([[-7.0, 5.0, 0.0, 7.0], [-5.0, 1.0, -8.0, 0.0], [-1.0, 2.0, -3.0, 5.0], [-3.0, 2.0, 0.0, 5.0]]), np.array([[1.0, 1.0, 1.0, 1.0], [-1.0, 2.0, -1.0, -1.0], [-1.0, 1.0, 3.0, 1.0], [-1.0, 1.0, -1.0, 4.0]]), np.array([[4.0, -1.0, 1.0, -1.0], [1.0, 3.0, -1.0, 1.0], [-1.0, 1.0, 2.0, -1.0], [1.0, -1.0, 1.0, 1.0]]))])
@pytest.mark.parametrize('dtype', REAL_DTYPES)
def test_tgsyl_NAG(a, b, c, d, e, f, rans, lans, dtype):
    atol = 0.0001
    tgsyl = get_lapack_funcs('tgsyl', dtype=dtype)
    rout, lout, scale, dif, info = tgsyl(a, b, c, d, e, f)
    assert_equal(info, 0)
    assert_allclose(scale, 1.0, rtol=0, atol=np.finfo(dtype).eps * 100, err_msg='SCALE must be 1.0')
    assert_allclose(dif, 0.0, rtol=0, atol=np.finfo(dtype).eps * 100, err_msg='DIF must be nearly 0')
    assert_allclose(rout, rans, atol=atol, err_msg='Solution for R is incorrect')
    assert_allclose(lout, lans, atol=atol, err_msg='Solution for L is incorrect')