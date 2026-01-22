import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
def test_mean_zero(self):
    x = np.array([10, -3, 1, -4, -4])
    y = variation(x)
    assert_equal(y, np.inf)
    x2 = np.array([x, -10 * x])
    y2 = variation(x2, axis=1)
    assert_equal(y2, [np.inf, np.inf])