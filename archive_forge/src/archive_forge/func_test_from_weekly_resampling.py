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
def test_from_weekly_resampling(self):
    idxh = date_range('1/1/1999', periods=52, freq='W')
    idxl = date_range('1/1/1999', periods=12, freq='ME')
    high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
    low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
    _, ax = mpl.pyplot.subplots()
    low.plot(ax=ax)
    high.plot(ax=ax)
    expected_h = idxh.to_period().asi8.astype(np.float64)
    expected_l = np.array([1514, 1519, 1523, 1527, 1531, 1536, 1540, 1544, 1549, 1553, 1558, 1562], dtype=np.float64)
    for line in ax.get_lines():
        assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq
        xdata = line.get_xdata(orig=False)
        if len(xdata) == 12:
            tm.assert_numpy_array_equal(xdata, expected_l)
        else:
            tm.assert_numpy_array_equal(xdata, expected_h)