from itertools import product, permutations
import numpy as np
from numpy.testing import assert_array_less, assert_allclose
from pytest import raises as assert_raises
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse._sputils import matrix
def test_orthogonal_procrustes_checkfinite_exception():
    np.random.seed(1234)
    m, n = (2, 3)
    A_good = np.random.randn(m, n)
    B_good = np.random.randn(m, n)
    for bad_value in (np.inf, -np.inf, np.nan):
        A_bad = A_good.copy()
        A_bad[1, 2] = bad_value
        B_bad = B_good.copy()
        B_bad[1, 2] = bad_value
        for A, B in ((A_good, B_bad), (A_bad, B_good), (A_bad, B_bad)):
            assert_raises(ValueError, orthogonal_procrustes, A, B)