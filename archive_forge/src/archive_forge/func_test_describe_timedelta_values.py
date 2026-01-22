import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_timedelta_values(self):
    t1 = pd.timedelta_range('1 days', freq='D', periods=5)
    t2 = pd.timedelta_range('1 hours', freq='h', periods=5)
    df = DataFrame({'t1': t1, 't2': t2})
    expected = DataFrame({'t1': [5, pd.Timedelta('3 days'), df.iloc[:, 0].std(), pd.Timedelta('1 days'), pd.Timedelta('2 days'), pd.Timedelta('3 days'), pd.Timedelta('4 days'), pd.Timedelta('5 days')], 't2': [5, pd.Timedelta('3 hours'), df.iloc[:, 1].std(), pd.Timedelta('1 hours'), pd.Timedelta('2 hours'), pd.Timedelta('3 hours'), pd.Timedelta('4 hours'), pd.Timedelta('5 hours')]}, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    result = df.describe()
    tm.assert_frame_equal(result, expected)
    exp_repr = '                              t1                         t2\ncount                          5                          5\nmean             3 days 00:00:00            0 days 03:00:00\nstd    1 days 13:56:50.394919273  0 days 01:34:52.099788303\nmin              1 days 00:00:00            0 days 01:00:00\n25%              2 days 00:00:00            0 days 02:00:00\n50%              3 days 00:00:00            0 days 03:00:00\n75%              4 days 00:00:00            0 days 04:00:00\nmax              5 days 00:00:00            0 days 05:00:00'
    assert repr(result) == exp_repr