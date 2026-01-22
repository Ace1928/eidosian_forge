from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_nsmallest_nlargest(self, s_main_dtypes_split):
    ser = s_main_dtypes_split
    tm.assert_series_equal(ser.nsmallest(2), ser.iloc[[2, 1]])
    tm.assert_series_equal(ser.nsmallest(2, keep='last'), ser.iloc[[2, 3]])
    empty = ser.iloc[0:0]
    tm.assert_series_equal(ser.nsmallest(0), empty)
    tm.assert_series_equal(ser.nsmallest(-1), empty)
    tm.assert_series_equal(ser.nlargest(0), empty)
    tm.assert_series_equal(ser.nlargest(-1), empty)
    tm.assert_series_equal(ser.nsmallest(len(ser)), ser.sort_values())
    tm.assert_series_equal(ser.nsmallest(len(ser) + 1), ser.sort_values())
    tm.assert_series_equal(ser.nlargest(len(ser)), ser.iloc[[4, 0, 1, 3, 2]])
    tm.assert_series_equal(ser.nlargest(len(ser) + 1), ser.iloc[[4, 0, 1, 3, 2]])