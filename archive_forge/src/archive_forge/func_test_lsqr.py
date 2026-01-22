import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
def test_lsqr(matrices):
    A_dense, A_sparse, b = matrices
    res0 = splin.lsqr(A_dense, b)
    res = splin.lsqr(A_sparse, b)
    assert_allclose(res[0], res0[0], atol=1e-05)