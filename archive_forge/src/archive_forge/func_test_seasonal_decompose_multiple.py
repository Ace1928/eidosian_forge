from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
def test_seasonal_decompose_multiple():
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809, 530, 489, 540, 457, 195, 176, 337, 239, 128, 102, 232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    x = np.c_[x, x]
    res = seasonal_decompose(x, period=4)
    assert_allclose(res.trend[:, 0], res.trend[:, 1])
    assert_allclose(res.seasonal[:, 0], res.seasonal[:, 1])
    assert_allclose(res.resid[:, 0], res.resid[:, 1])