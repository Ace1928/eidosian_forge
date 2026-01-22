from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_datetime_tz(self):
    dates1 = pd.DatetimeIndex(['2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00', '2011-07-19 07:00:00', '2011-07-19 08:00:00', '2011-07-19 09:00:00'], dtype='M8[ns, US/Pacific]', name='dt1')
    dates2 = pd.DatetimeIndex(['2013-01-01 15:00:00', '2013-01-01 15:00:00', '2013-01-01 15:00:00', '2013-02-01 15:00:00', '2013-02-01 15:00:00', '2013-02-01 15:00:00'], dtype='M8[ns, Asia/Tokyo]')
    df = DataFrame({'label': ['a', 'a', 'a', 'b', 'b', 'b'], 'dt1': dates1, 'dt2': dates2, 'value1': np.arange(6, dtype='int64'), 'value2': [1, 2] * 3})
    exp_idx = dates1[:3]
    exp_col1 = Index(['value1', 'value1'])
    exp_col2 = Index(['a', 'b'], name='label')
    exp_col = MultiIndex.from_arrays([exp_col1, exp_col2])
    expected = DataFrame([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]], index=exp_idx, columns=exp_col)
    result = pivot_table(df, index=['dt1'], columns=['label'], values=['value1'])
    tm.assert_frame_equal(result, expected)
    exp_col1 = Index(['sum', 'sum', 'sum', 'sum', 'mean', 'mean', 'mean', 'mean'])
    exp_col2 = Index(['value1', 'value1', 'value2', 'value2'] * 2)
    exp_col3 = pd.DatetimeIndex(['2013-01-01 15:00:00', '2013-02-01 15:00:00'] * 4, dtype='M8[ns, Asia/Tokyo]', name='dt2')
    exp_col = MultiIndex.from_arrays([exp_col1, exp_col2, exp_col3])
    expected1 = DataFrame(np.array([[0, 3, 1, 2], [1, 4, 2, 1], [2, 5, 1, 2]], dtype='int64'), index=exp_idx, columns=exp_col[:4])
    expected2 = DataFrame(np.array([[0.0, 3.0, 1.0, 2.0], [1.0, 4.0, 2.0, 1.0], [2.0, 5.0, 1.0, 2.0]]), index=exp_idx, columns=exp_col[4:])
    expected = concat([expected1, expected2], axis=1)
    result = pivot_table(df, index=['dt1'], columns=['dt2'], values=['value1', 'value2'], aggfunc=['sum', 'mean'])
    tm.assert_frame_equal(result, expected)