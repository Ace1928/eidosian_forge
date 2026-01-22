from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
def test_where_index_period(self):
    dti = pd.date_range('2016-01-01', periods=3, freq='QS')
    pi = dti.to_period('Q')
    cond = np.array([False, True, False])
    value = pi[-1] + pi.freq * 10
    expected = pd.PeriodIndex([value, pi[1], value])
    result = pi.where(cond, value)
    tm.assert_index_equal(result, expected)
    other = np.asarray(pi + pi.freq * 10, dtype=object)
    result = pi.where(cond, other)
    expected = pd.PeriodIndex([other[0], pi[1], other[2]])
    tm.assert_index_equal(result, expected)
    td = pd.Timedelta(days=4)
    expected = pd.Index([td, pi[1], td], dtype=object)
    result = pi.where(cond, td)
    tm.assert_index_equal(result, expected)
    per = pd.Period('2020-04-21', 'D')
    expected = pd.Index([per, pi[1], per], dtype=object)
    result = pi.where(cond, per)
    tm.assert_index_equal(result, expected)