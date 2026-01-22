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
def test_unstack_period_frame(self):
    idx1 = pd.PeriodIndex(['2014-01', '2014-02', '2014-02', '2014-02', '2014-01', '2014-01'], freq='M', name='period1')
    idx2 = pd.PeriodIndex(['2013-12', '2013-12', '2014-02', '2013-10', '2013-10', '2014-02'], freq='M', name='period2')
    value = {'A': [1, 2, 3, 4, 5, 6], 'B': [6, 5, 4, 3, 2, 1]}
    idx = MultiIndex.from_arrays([idx1, idx2])
    df = DataFrame(value, index=idx)
    result1 = df.unstack()
    result2 = df.unstack(level=1)
    result3 = df.unstack(level=0)
    e_1 = pd.PeriodIndex(['2014-01', '2014-02'], freq='M', name='period1')
    e_2 = pd.PeriodIndex(['2013-10', '2013-12', '2014-02', '2013-10', '2013-12', '2014-02'], freq='M', name='period2')
    e_cols = MultiIndex.from_arrays(['A A A B B B'.split(), e_2])
    expected = DataFrame([[5, 1, 6, 2, 6, 1], [4, 2, 3, 3, 5, 4]], index=e_1, columns=e_cols)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    e_1 = pd.PeriodIndex(['2014-01', '2014-02', '2014-01', '2014-02'], freq='M', name='period1')
    e_2 = pd.PeriodIndex(['2013-10', '2013-12', '2014-02'], freq='M', name='period2')
    e_cols = MultiIndex.from_arrays(['A A B B'.split(), e_1])
    expected = DataFrame([[5, 4, 2, 3], [1, 2, 6, 5], [6, 3, 1, 4]], index=e_2, columns=e_cols)
    tm.assert_frame_equal(result3, expected)