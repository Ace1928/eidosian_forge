from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
def test_filt(self):
    filt = np.array([1 / 8.0, 1 / 4.0, 1.0 / 4, 1 / 4.0, 1 / 8.0])
    res_add = seasonal_decompose(self.data.values, filt=filt, period=4)
    assert_almost_equal(res_add.seasonal, SEASONAL, 2)
    assert_almost_equal(res_add.trend, TREND, 2)
    assert_almost_equal(res_add.resid, RANDOM, 3)