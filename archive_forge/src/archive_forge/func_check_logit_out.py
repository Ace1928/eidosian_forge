import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.special import logit, expit, log_expit
def check_logit_out(self, dtype, expected):
    a = np.linspace(0, 1, 10)
    a = np.array(a, dtype=dtype)
    with np.errstate(divide='ignore'):
        actual = logit(a)
    assert_almost_equal(actual, expected)
    assert_equal(actual.dtype, np.dtype(dtype))