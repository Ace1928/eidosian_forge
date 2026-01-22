import numpy as np
from numpy.testing import assert_equal, assert_raises
from statsmodels.tsa.arima.tools import (
def test_standardize_lag_order_list_bool():
    assert_equal(standardize_lag_order([0]), 0)
    assert_equal(standardize_lag_order([1]), 1)
    assert_equal(standardize_lag_order([0, 1]), [2])
    assert_equal(standardize_lag_order([0, 1, 0, 1]), [2, 4])