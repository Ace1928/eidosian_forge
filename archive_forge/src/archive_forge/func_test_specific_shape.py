import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
def test_specific_shape(self):
    n = 6
    mik = MikotaPair(n)
    mik_k = mik.k
    mik_m = mik.m
    assert_array_equal(mik_k.toarray(), mik_k(np.eye(n)))
    assert_array_equal(mik_m.toarray(), mik_m(np.eye(n)))
    k = np.array([[11, -5, 0, 0, 0, 0], [-5, 9, -4, 0, 0, 0], [0, -4, 7, -3, 0, 0], [0, 0, -3, 5, -2, 0], [0, 0, 0, -2, 3, -1], [0, 0, 0, 0, -1, 1]])
    np.array_equal(k, mik_k.toarray())
    np.array_equal(mik_k.tosparse().toarray(), k)
    kb = np.array([[0, -5, -4, -3, -2, -1], [11, 9, 7, 5, 3, 1]])
    np.array_equal(kb, mik_k.tobanded())
    minv = np.arange(1, n + 1)
    np.array_equal(np.diag(1.0 / minv), mik_m.toarray())
    np.array_equal(mik_m.tosparse().toarray(), mik_m.toarray())
    np.array_equal(1.0 / minv, mik_m.tobanded())
    e = np.array([1, 4, 9, 16, 25, 36])
    np.array_equal(e, mik.eigenvalues())
    np.array_equal(e[:2], mik.eigenvalues(2))