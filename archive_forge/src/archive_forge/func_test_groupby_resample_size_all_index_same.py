from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_groupby_resample_size_all_index_same():
    df = DataFrame({'A': [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3, 'B': np.arange(12)}, index=date_range('31/12/2000 18:00', freq='h', periods=12))
    msg = 'DataFrameGroupBy.resample operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').resample('D').size()
    mi_exp = pd.MultiIndex.from_arrays([[1, 1, 2, 2], pd.DatetimeIndex(['2000-12-31', '2001-01-01'] * 2, dtype='M8[ns]')], names=['A', None])
    expected = Series(3, index=mi_exp)
    tm.assert_series_equal(result, expected)