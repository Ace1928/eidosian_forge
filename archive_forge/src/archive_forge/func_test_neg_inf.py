import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
def test_neg_inf(self):
    x1 = np.array([-3, -5])
    assert_equal(variation(x1, ddof=2), -np.inf)
    x2 = np.array([[np.nan, 1, -10, np.nan], [-20, -3, np.nan, np.nan]])
    assert_equal(variation(x2, axis=1, ddof=2, nan_policy='omit'), [-np.inf, -np.inf])