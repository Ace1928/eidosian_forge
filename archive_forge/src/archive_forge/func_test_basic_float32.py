import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.special import logit, expit, log_expit
def test_basic_float32(self):
    x = np.array([-32, -20, -10, -3, -1, -0.1, -1e-09, 0, 1e-09, 0.1, 1, 10, 100], dtype=np.float32)
    y = log_expit(x)
    expected = np.array([-32.0, -20.0, -10.000046, -3.0485873, -1.3132616, -0.7443967, -0.6931472, -0.6931472, -0.6931472, -0.64439666, -0.3132617, -4.5398898e-05, -3.8e-44], dtype=np.float32)
    assert_allclose(y, expected, rtol=5e-07)