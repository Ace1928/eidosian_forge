from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_logical_operators_bool_dtype_with_empty(self):
    index = list('bca')
    s_tft = Series([True, False, True], index=index)
    s_fff = Series([False, False, False], index=index)
    s_empty = Series([], dtype=object)
    res = s_tft & s_empty
    expected = s_fff.sort_index()
    tm.assert_series_equal(res, expected)
    res = s_tft | s_empty
    expected = s_tft.sort_index()
    tm.assert_series_equal(res, expected)