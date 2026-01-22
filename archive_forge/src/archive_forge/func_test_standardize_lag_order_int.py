import numpy as np
from numpy.testing import assert_equal, assert_raises
from statsmodels.tsa.arima.tools import (
def test_standardize_lag_order_int():
    assert_equal(standardize_lag_order(0, title='test'), 0)
    assert_equal(standardize_lag_order(3), 3)