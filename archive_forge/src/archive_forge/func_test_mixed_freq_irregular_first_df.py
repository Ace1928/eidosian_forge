from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props
from pandas.tseries.offsets import WeekOfMonth
def test_mixed_freq_irregular_first_df(self):
    s1 = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20)).to_frame()
    s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
    _, ax = mpl.pyplot.subplots()
    s2.plot(style='g', ax=ax)
    s1.plot(ax=ax)
    assert not hasattr(ax, 'freq')
    lines = ax.get_lines()
    x1 = lines[0].get_xdata()
    tm.assert_numpy_array_equal(x1, s2.index.astype(object).values)
    x2 = lines[1].get_xdata()
    tm.assert_numpy_array_equal(x2, s1.index.astype(object).values)