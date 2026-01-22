import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_slice_step_ne1(self, series_with_interval_index):
    ser = series_with_interval_index.copy()
    expected = ser.iloc[0:4:2]
    result = ser[0:4:2]
    tm.assert_series_equal(result, expected)
    result2 = ser[0:4][::2]
    tm.assert_series_equal(result2, expected)