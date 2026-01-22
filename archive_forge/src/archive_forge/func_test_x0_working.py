import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_x0_working(solver):
    rng = np.random.default_rng(1685363802304750)
    n = 10
    A = rng.random(size=[n, n])
    A = A @ A.T
    b = rng.random(n)
    x0 = rng.random(n)
    if solver is minres:
        kw = dict(rtol=1e-06)
    else:
        kw = dict(atol=0, rtol=1e-06)
    x, info = solver(A, b, **kw)
    assert info == 0
    assert norm(A @ x - b) <= 1e-06 * norm(b)
    x, info = solver(A, b, x0=x0, **kw)
    assert info == 0
    assert norm(A @ x - b) <= 2e-06 * norm(b)