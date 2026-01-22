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
@pytest.mark.parametrize('A,sva_expect,u_expect,v_expect', [(np.array([[2.27, -1.54, 1.15, -1.94], [0.28, -1.67, 0.94, -0.78], [-0.48, -3.09, 0.99, -0.21], [1.07, 1.22, 0.79, 0.63], [-2.35, 2.93, -1.45, 2.3], [0.62, -7.39, 1.03, -2.57]]), np.array([9.9966, 3.6831, 1.3569, 0.5]), np.array([[0.2774, -0.6003, -0.1277, 0.1323], [0.202, -0.0301, 0.2805, 0.7034], [0.2918, 0.3348, 0.6453, 0.1906], [-0.0938, -0.3699, 0.6781, -0.5399], [-0.4213, 0.5266, 0.0413, -0.0575], [0.7816, 0.3353, -0.1645, -0.3957]]), np.array([[0.1921, -0.803, 0.0041, -0.5642], [-0.8794, -0.3926, -0.0752, 0.2587], [0.214, -0.298, 0.7827, 0.5027], [-0.3795, 0.3351, 0.6178, -0.6017]]))])
def test_gejsv_NAG(A, sva_expect, u_expect, v_expect):
    """
    This test implements the example found in the NAG manual, f08khf.
    An example was not found for the complex case.
    """
    atol = 0.0001
    gejsv = get_lapack_funcs('gejsv', dtype=A.dtype)
    sva, u, v, work, iwork, info = gejsv(A)
    assert_allclose(sva_expect, sva, atol=atol)
    assert_allclose(u_expect, u, atol=atol)
    assert_allclose(v_expect, v, atol=atol)