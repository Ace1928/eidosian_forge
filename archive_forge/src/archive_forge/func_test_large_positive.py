import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.special import logit, expit, log_expit
def test_large_positive(self):
    x = np.array([750.0, 1000.0, 10000.0])
    y = log_expit(x)
    assert_equal(y, np.array([-0.0, -0.0, -0.0]))