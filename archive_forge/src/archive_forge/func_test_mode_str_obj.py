from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dropna, expected1, expected2, expected3', [(True, ['b'], ['bar'], ['nan']), (False, ['b'], [np.nan], ['nan'])])
def test_mode_str_obj(self, dropna, expected1, expected2, expected3):
    data = ['a'] * 2 + ['b'] * 3
    s = Series(data, dtype='c')
    result = s.mode(dropna)
    expected1 = Series(expected1, dtype='c')
    tm.assert_series_equal(result, expected1)
    data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]
    s = Series(data, dtype=object)
    result = s.mode(dropna)
    expected2 = Series(expected2, dtype=None if expected2 == ['bar'] else object)
    tm.assert_series_equal(result, expected2)
    data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]
    s = Series(data, dtype=object).astype(str)
    result = s.mode(dropna)
    expected3 = Series(expected3)
    tm.assert_series_equal(result, expected3)