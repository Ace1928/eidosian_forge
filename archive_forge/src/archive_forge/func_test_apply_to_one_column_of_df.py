from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_apply_to_one_column_of_df():
    df = DataFrame({'col': range(10), 'col1': range(10, 20)}, index=date_range('2012-01-01', periods=10, freq='20min'))
    result = df.resample('h').apply(lambda group: group.col.sum())
    expected = Series([3, 12, 21, 9], index=date_range('2012-01-01', periods=4, freq='h'))
    tm.assert_series_equal(result, expected)
    result = df.resample('h').apply(lambda group: group['col'].sum())
    tm.assert_series_equal(result, expected)