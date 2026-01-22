import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
def test_rank_estimates_lin_op(self, A):
    B = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=A.dtype)
    for M in [A, B]:
        ML = aslinearoperator(M)
        rank_tol = 1e-09
        rank_np = np.linalg.matrix_rank(M, norm(M, 2) * rank_tol)
        rank_est = pymatrixid.estimate_rank(ML, rank_tol)
        assert_(rank_est >= rank_np - 4)
        assert_(rank_est <= rank_np + 4)