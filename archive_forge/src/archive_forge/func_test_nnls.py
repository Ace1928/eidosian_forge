import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.optimize import nnls
def test_nnls(self):
    a = np.arange(25.0).reshape(-1, 5)
    x = np.arange(5.0)
    y = a @ x
    x, res = nnls(a, y)
    assert res < 1e-07
    assert np.linalg.norm(a @ x - y) < 1e-07