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
def test_unstack_period_series(self):
    idx1 = pd.PeriodIndex(['2013-01', '2013-01', '2013-02', '2013-02', '2013-03', '2013-03'], freq='M', name='period')
    idx2 = Index(['A', 'B'] * 3, name='str')
    value = [1, 2, 3, 4, 5, 6]
    idx = MultiIndex.from_arrays([idx1, idx2])
    s = Series(value, index=idx)
    result1 = s.unstack()
    result2 = s.unstack(level=1)
    result3 = s.unstack(level=0)
    e_idx = pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period')
    expected = DataFrame({'A': [1, 3, 5], 'B': [2, 4, 6]}, index=e_idx, columns=['A', 'B'])
    expected.columns.name = 'str'
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected.T)
    idx1 = pd.PeriodIndex(['2013-01', '2013-01', '2013-02', '2013-02', '2013-03', '2013-03'], freq='M', name='period1')
    idx2 = pd.PeriodIndex(['2013-12', '2013-11', '2013-10', '2013-09', '2013-08', '2013-07'], freq='M', name='period2')
    idx = MultiIndex.from_arrays([idx1, idx2])
    s = Series(value, index=idx)
    result1 = s.unstack()
    result2 = s.unstack(level=1)
    result3 = s.unstack(level=0)
    e_idx = pd.PeriodIndex(['2013-01', '2013-02', '2013-03'], freq='M', name='period1')
    e_cols = pd.PeriodIndex(['2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12'], freq='M', name='period2')
    expected = DataFrame([[np.nan, np.nan, np.nan, np.nan, 2, 1], [np.nan, np.nan, 4, 3, np.nan, np.nan], [6, 5, np.nan, np.nan, np.nan, np.nan]], index=e_idx, columns=e_cols)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected.T)