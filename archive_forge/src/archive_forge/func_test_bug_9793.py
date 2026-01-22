import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('rand', [True, False])
@pytest.mark.parametrize('eps', [1, 0.1])
def test_bug_9793(self, dtype, rand, eps):
    if _IS_32BIT and dtype == np.complex128 and rand:
        pytest.xfail('bug in external fortran code')
    A = np.array([[-1, -1, -1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]], dtype=dtype, order='C')
    B = A.copy()
    interp_decomp(A.T, eps, rand=rand)
    assert_array_equal(A, B)