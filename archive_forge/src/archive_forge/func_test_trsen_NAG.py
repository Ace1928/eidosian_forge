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
@pytest.mark.parametrize('t, q, expect, select, expect_s, expect_sep', [(np.array([[0.7995, -0.1144, 0.006, 0.0336], [0.0, -0.0994, 0.2478, 0.3474], [0.0, -0.6483, -0.0994, 0.2026], [0.0, 0.0, 0.0, -0.1007]]), np.array([[0.6551, 0.1037, 0.345, 0.6641], [0.5236, -0.5807, -0.6141, -0.1068], [-0.5362, -0.3073, -0.2935, 0.7293], [0.0956, 0.7467, -0.6463, 0.1249]]), np.array([[0.35, 0.45, -0.14, -0.17], [0.09, 0.07, -0.5399, 0.35], [-0.44, -0.33, -0.03, 0.17], [0.25, -0.32, -0.13, 0.11]]), np.array([1, 0, 0, 1]), 1.75, 3.22), (np.array([[-6.0004 - 6.9999j, 0.3637 - 0.3656j, -0.188 + 0.4787j, 0.8785 - 0.2539j], [0.0 + 0j, -5.0 + 2.006j, -0.0307 - 0.7217j, -0.229 + 0.1313j], [0.0 + 0j, 0.0 + 0j, 7.9982 - 0.9964j, 0.9357 + 0.5359j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 3.0023 - 3.9998j]]), np.array([[-0.8347 - 0.1364j, -0.0628 + 0.3806j, 0.2765 - 0.0846j, 0.0633 - 0.2199j], [0.0664 - 0.2968j, 0.2365 + 0.524j, -0.5877 - 0.4208j, 0.0835 + 0.2183j], [-0.0362 - 0.3215j, 0.3143 - 0.5473j, 0.0576 - 0.5736j, 0.0057 - 0.4058j], [0.0086 + 0.2958j, -0.3416 - 0.0757j, -0.19 - 0.16j, 0.8327 - 0.1868j]]), np.array([[-3.9702 - 5.0406j, -4.1108 + 3.7002j, -0.3403 + 1.0098j, 1.2899 - 0.859j], [0.3397 - 1.5006j, 1.5201 - 0.4301j, 1.8797 - 5.3804j, 3.3606 + 0.6498j], [3.3101 - 3.8506j, 2.4996 + 3.4504j, 0.8802 - 1.0802j, 0.6401 - 1.48j], [-1.0999 + 0.8199j, 1.8103 - 1.5905j, 3.2502 + 1.3297j, 1.5701 - 3.4397j]]), np.array([1, 0, 0, 1]), 1.02, 0.182)])
def test_trsen_NAG(t, q, select, expect, expect_s, expect_sep):
    """
    This test implements the example found in the NAG manual,
    f08qgc, f08quc.
    """
    atol = 0.0001
    atol2 = 0.01
    trsen, trsen_lwork = get_lapack_funcs(('trsen', 'trsen_lwork'), dtype=t.dtype)
    lwork = _compute_lwork(trsen_lwork, select, t)
    if t.dtype in COMPLEX_DTYPES:
        result = trsen(select, t, q, lwork=lwork)
    else:
        result = trsen(select, t, q, lwork=lwork, liwork=lwork[1])
    assert_equal(result[-1], 0)
    t = result[0]
    q = result[1]
    if t.dtype in COMPLEX_DTYPES:
        s = result[4]
        sep = result[5]
    else:
        s = result[5]
        sep = result[6]
    assert_allclose(expect, q @ t @ q.conj().T, atol=atol)
    assert_allclose(expect_s, 1 / s, atol=atol2)
    assert_allclose(expect_sep, 1 / sep, atol=atol2)