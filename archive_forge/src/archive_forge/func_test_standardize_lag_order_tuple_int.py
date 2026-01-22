import numpy as np
from numpy.testing import assert_equal, assert_raises
from statsmodels.tsa.arima.tools import (
def test_standardize_lag_order_tuple_int():
    assert_equal(standardize_lag_order((1, 2)), 2)
    assert_equal(standardize_lag_order((1, 3)), [1, 3])