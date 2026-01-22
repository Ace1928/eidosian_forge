from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b
def testInitialization(self):
    x_ref, _, itn_ref, normr_ref, *_ = lsmr(G, b)
    assert_allclose(norm(b - G @ x_ref), normr_ref, atol=1e-06)
    x0 = zeros(b.shape)
    x = lsmr(G, b, x0=x0)[0]
    assert_allclose(x, x_ref)
    x0 = lsmr(G, b, maxiter=1)[0]
    x, _, itn, normr, *_ = lsmr(G, b, x0=x0)
    assert_allclose(norm(b - G @ x), normr, atol=1e-06)
    assert itn - itn_ref in (0, 1)
    assert normr < normr_ref * (1 + 1e-06)