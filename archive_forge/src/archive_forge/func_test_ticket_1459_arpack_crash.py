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
def test_ticket_1459_arpack_crash():
    for dtype in [np.float32, np.float64]:
        N = 6
        k = 2
        np.random.seed(2301)
        A = np.random.random((N, N)).astype(dtype)
        v0 = np.array([-0.7106356825890785, -0.8318511179572923, -0.343659253822274, 0.4612253368455228, -0.5800134111596904, -0.07884487757008429], dtype=dtype)
        evals, evecs = eigs(A, k, v0=v0)