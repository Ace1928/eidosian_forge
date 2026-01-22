import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
def test_propagate_nan(self):
    a = np.arange(8).reshape(2, -1).astype(float)
    a[1, 0] = np.nan
    v = variation(a, axis=1, nan_policy='propagate')
    assert_allclose(v, [np.sqrt(5 / 4) / 1.5, np.nan], atol=1e-15)