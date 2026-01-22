from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('method, unit', [('sum', 0), ('prod', 1)])
@pytest.mark.parametrize('numeric_only', [None, True, False])
def test_sum_prod_nanops(self, method, unit, numeric_only):
    idx = ['a', 'b', 'c']
    df = DataFrame({'a': [unit, unit], 'b': [unit, np.nan], 'c': [np.nan, np.nan]})
    result = getattr(df, method)(numeric_only=numeric_only)
    expected = Series([unit, unit, unit], index=idx, dtype='float64')
    tm.assert_series_equal(result, expected)
    result = getattr(df, method)(numeric_only=numeric_only, min_count=1)
    expected = Series([unit, unit, np.nan], index=idx)
    tm.assert_series_equal(result, expected)
    result = getattr(df, method)(numeric_only=numeric_only, min_count=0)
    expected = Series([unit, unit, unit], index=idx, dtype='float64')
    tm.assert_series_equal(result, expected)
    result = getattr(df.iloc[1:], method)(numeric_only=numeric_only, min_count=1)
    expected = Series([unit, np.nan, np.nan], index=idx)
    tm.assert_series_equal(result, expected)
    df = DataFrame({'A': [unit] * 10, 'B': [unit] * 5 + [np.nan] * 5})
    result = getattr(df, method)(numeric_only=numeric_only, min_count=5)
    expected = Series(result, index=['A', 'B'])
    tm.assert_series_equal(result, expected)
    result = getattr(df, method)(numeric_only=numeric_only, min_count=6)
    expected = Series(result, index=['A', 'B'])
    tm.assert_series_equal(result, expected)