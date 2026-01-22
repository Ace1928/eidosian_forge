from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_dtypes(self, using_infer_string):
    rows = [[1, 1, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4], [2, 2, 3, 4]]
    df = DataFrame(rows, columns=list('ABCD'))
    result = df.dtypes
    expected = Series([np.dtype('int64')] * 4, index=list('ABCD'))
    tm.assert_series_equal(result, expected)
    df2 = df.set_index(['A', 'B'])
    df3 = df2.unstack('B')
    result = df3.dtypes
    expected = Series([np.dtype('int64')] * 4, index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B')))
    tm.assert_series_equal(result, expected)
    df2 = df.set_index(['A', 'B'])
    df2['C'] = 3.0
    df3 = df2.unstack('B')
    result = df3.dtypes
    expected = Series([np.dtype('float64')] * 2 + [np.dtype('int64')] * 2, index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B')))
    tm.assert_series_equal(result, expected)
    df2['D'] = 'foo'
    df3 = df2.unstack('B')
    result = df3.dtypes
    dtype = 'string' if using_infer_string else np.dtype('object')
    expected = Series([np.dtype('float64')] * 2 + [dtype] * 2, index=MultiIndex.from_arrays([['C', 'C', 'D', 'D'], [1, 2, 1, 2]], names=(None, 'B')))
    tm.assert_series_equal(result, expected)