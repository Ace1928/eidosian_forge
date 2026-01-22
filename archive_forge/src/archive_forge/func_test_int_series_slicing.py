from datetime import (
import numpy as np
import pytest
from pandas.errors import UnsortedIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
def test_int_series_slicing(self, multiindex_year_month_day_dataframe_random_data):
    ymd = multiindex_year_month_day_dataframe_random_data
    s = ymd['A']
    result = s[5:]
    expected = s.reindex(s.index[5:])
    tm.assert_series_equal(result, expected)
    s = ymd['A'].copy()
    exp = ymd['A'].copy()
    s[5:] = 0
    exp.iloc[5:] = 0
    tm.assert_numpy_array_equal(s.values, exp.values)
    result = ymd[5:]
    expected = ymd.reindex(s.index[5:])
    tm.assert_frame_equal(result, expected)