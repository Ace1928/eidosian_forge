from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_equality_comparison_broadcasts_over_array(self):
    interval = Interval(0, 1)
    arr = np.array([interval, interval])
    result = interval == arr
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)