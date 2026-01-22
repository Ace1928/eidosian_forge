import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy.optimize import nnls
def test_nnls_wide(self):
    a = self.rng.uniform(low=-10, high=10, size=[100, 120])
    x = np.abs(self.rng.uniform(low=-2, high=2, size=[120]))
    x[::2] = 0
    b = a @ x
    xact, rnorm = nnls(a, b, atol=500 * np.linalg.norm(a, 1) * np.spacing(1.0))
    assert_allclose(xact, x, rtol=0.0, atol=1e-10)
    assert rnorm < 1e-12