import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
def test_full_rank(self):
    eps = 1e-12
    A = np.random.rand(16, 8)
    k, idx, proj = pymatrixid.interp_decomp(A, eps)
    assert_equal(k, A.shape[1])
    P = pymatrixid.reconstruct_interp_matrix(idx, proj)
    B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
    assert_allclose(A, B @ P)
    idx, proj = pymatrixid.interp_decomp(A, k)
    P = pymatrixid.reconstruct_interp_matrix(idx, proj)
    B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
    assert_allclose(A, B @ P)