from io import BytesIO
from itertools import product
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingWLS, RollingOLS
def test_has_nan(data):
    y, x, w = data
    mod = RollingWLS(y, x, window=100, weights=w)
    has_nan = np.zeros(y.shape[0], dtype=bool)
    for i in range(100, y.shape[0] + 1):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        has_nan[i - 1] = np.squeeze(np.any(np.isnan(_y)) or np.any(np.isnan(_x)))
    assert_array_equal(mod._has_nan, has_nan)