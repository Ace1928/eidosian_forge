import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_loc_with_slices(self, series_with_interval_index, indexer_sl):
    ser = series_with_interval_index.copy()
    expected = ser.iloc[:3]
    result = indexer_sl(ser)[Interval(0, 1):Interval(2, 3)]
    tm.assert_series_equal(expected, result)
    expected = ser.iloc[3:]
    result = indexer_sl(ser)[Interval(3, 4):]
    tm.assert_series_equal(expected, result)
    msg = 'Interval objects are not currently supported'
    with pytest.raises(NotImplementedError, match=msg):
        indexer_sl(ser)[Interval(3, 6):]
    with pytest.raises(NotImplementedError, match=msg):
        indexer_sl(ser)[Interval(3, 4, closed='left'):]