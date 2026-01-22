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
def test_breakdown_underdetermined(self):
    A = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=float)
    bs = [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 0]), np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])]
    for b in bs:
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, '.*called without specifying.*')
            xp, info = lgmres(A, b, maxiter=1)
        resp = np.linalg.norm(A.dot(xp) - b)
        K = np.c_[b, A.dot(b), A.dot(A.dot(b)), A.dot(A.dot(A.dot(b)))]
        y, _, _, _ = np.linalg.lstsq(A.dot(K), b, rcond=-1)
        x = K.dot(y)
        res = np.linalg.norm(A.dot(x) - b)
        assert_allclose(resp, res, err_msg=repr(b))