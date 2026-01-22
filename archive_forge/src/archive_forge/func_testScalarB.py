from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b
def testScalarB(self):
    A = array([[1.0, 2.0]])
    b = 3.0
    x = lsmr(A, b)[0]
    assert norm(A.dot(x) - b) == pytest.approx(0)