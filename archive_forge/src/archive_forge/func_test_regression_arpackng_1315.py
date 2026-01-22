import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def test_regression_arpackng_1315():
    for dtype in [np.float32, np.float64]:
        np.random.seed(1234)
        w0 = np.arange(1, 1000 + 1).astype(dtype)
        A = diags([w0], [0], shape=(1000, 1000))
        v0 = np.random.rand(1000).astype(dtype)
        w, v = eigs(A, k=9, ncv=2 * 9 + 1, which='LM', v0=v0)
        assert_allclose(np.sort(w), np.sort(w0[-9:]), rtol=0.0001)