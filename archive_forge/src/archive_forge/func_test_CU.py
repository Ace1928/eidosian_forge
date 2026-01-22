from numpy.testing import (assert_, assert_allclose, assert_equal,
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import gcrotmk, gmres
def test_CU(self):
    for discard_C in (True, False):
        CU = []
        x0, count_0 = do_solve(CU=CU, discard_C=discard_C)
        assert_(len(CU) > 0)
        assert_(len(CU) <= 6)
        if discard_C:
            for c, u in CU:
                assert_(c is None)
        x1, count_1 = do_solve(CU=CU, discard_C=discard_C)
        if discard_C:
            assert_equal(count_1, 2 + len(CU))
        else:
            assert_equal(count_1, 3)
        assert_(count_1 <= count_0 / 2)
        assert_allclose(x1, x0, atol=1e-14)