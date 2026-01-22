import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_allclose, assert_
from scipy.sparse.linalg._isolve import minres
from pytest import raises as assert_raises
def test_minres_precond_exact_x0():
    np.random.seed(1234)
    rtol = 1e-06
    a = np.eye(10)
    b = np.ones(10)
    c = np.ones(10)
    m = np.random.randn(10, 10)
    m = np.dot(m, m.T)
    x = minres(a, b, M=m, x0=c, rtol=rtol)[0]
    assert norm(a @ x - b) <= rtol * norm(b)