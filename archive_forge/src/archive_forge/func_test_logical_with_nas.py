import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
def test_logical_with_nas(self):
    d = DataFrame({'a': [np.nan, False], 'b': [True, True]})
    result = d['a'] | d['b']
    expected = Series([False, True])
    tm.assert_series_equal(result, expected)
    result = d['a'].fillna(False) | d['b']
    expected = Series([True, True])
    tm.assert_series_equal(result, expected)
    msg = "The 'downcast' keyword in fillna is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = d['a'].fillna(False, downcast=False) | d['b']
    expected = Series([True, True])
    tm.assert_series_equal(result, expected)