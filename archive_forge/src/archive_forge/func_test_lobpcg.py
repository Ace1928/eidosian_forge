import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
def test_lobpcg(matrices):
    A_dense, A_sparse, x = matrices
    X = x[:, None]
    w_dense, v_dense = splin.lobpcg(A_dense, X)
    w, v = splin.lobpcg(A_sparse, X)
    assert_allclose(w, w_dense)
    assert_allclose(v, v_dense)