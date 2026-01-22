import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
@pytest.mark.parametrize('rand,lin_op', [(False, False), (True, False), (True, True)])
def test_svd_fixed_precison(self, A, L, eps, rand, lin_op):
    if _IS_32BIT and A.dtype == np.complex128 and rand:
        pytest.xfail('bug in external fortran code')
    A_or_L = A if not lin_op else L
    U, S, V = pymatrixid.svd(A_or_L, eps, rand=rand)
    B = U * S @ V.T.conj()
    assert_allclose(A, B, rtol=eps, atol=1e-08)