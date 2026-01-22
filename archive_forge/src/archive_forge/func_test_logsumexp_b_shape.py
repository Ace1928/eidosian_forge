import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from scipy.special import logsumexp, softmax
def test_logsumexp_b_shape():
    a = np.zeros((4, 1, 2, 1))
    b = np.ones((3, 1, 5))
    logsumexp(a, b=b)