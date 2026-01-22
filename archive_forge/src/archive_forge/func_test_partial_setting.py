import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_setting(self):
    s_orig = Series([1, 2, 3])
    s = s_orig.copy()
    s[5] = 5
    expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
    tm.assert_series_equal(s, expected)
    s = s_orig.copy()
    s.loc[5] = 5
    expected = Series([1, 2, 3, 5], index=[0, 1, 2, 5])
    tm.assert_series_equal(s, expected)
    s = s_orig.copy()
    s[5] = 5.0
    expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
    tm.assert_series_equal(s, expected)
    s = s_orig.copy()
    s.loc[5] = 5.0
    expected = Series([1, 2, 3, 5.0], index=[0, 1, 2, 5])
    tm.assert_series_equal(s, expected)
    s = s_orig.copy()
    msg = 'iloc cannot enlarge its target object'
    with pytest.raises(IndexError, match=msg):
        s.iloc[3] = 5.0
    msg = 'index 3 is out of bounds for axis 0 with size 3'
    with pytest.raises(IndexError, match=msg):
        s.iat[3] = 5.0