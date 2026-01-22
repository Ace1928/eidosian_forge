import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_nearest():
    s = Series(np.arange(10, dtype='int64'))
    target = [0.1, 0.9, 1.5, 2.0]
    result = s.reindex(target, method='nearest')
    expected = Series(np.around(target).astype('int64'), target)
    tm.assert_series_equal(expected, result)
    result = s.reindex(target, method='nearest', tolerance=0.2)
    expected = Series([0, 1, np.nan, 2], target)
    tm.assert_series_equal(expected, result)
    result = s.reindex(target, method='nearest', tolerance=[0.3, 0.01, 0.4, 3])
    expected = Series([0, np.nan, np.nan, 2], target)
    tm.assert_series_equal(expected, result)