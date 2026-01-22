from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from platform import python_implementation
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import lgmres, gmres
def test_breakdown_with_outer_v(self):
    A = np.array([[1, 2], [3, 4]], dtype=float)
    b = np.array([1, 2])
    x = np.linalg.solve(A, b)
    v0 = np.array([1, 0])
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, '.*called without specifying.*')
        xp, info = lgmres(A, b, outer_v=[(v0, None), (x, None)], maxiter=1)
    assert_allclose(xp, x, atol=1e-12)