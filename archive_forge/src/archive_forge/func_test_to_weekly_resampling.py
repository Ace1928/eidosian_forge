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
def test_to_weekly_resampling(self):
    idxh = date_range('1/1/1999', periods=52, freq='W')
    idxl = date_range('1/1/1999', periods=12, freq='ME')
    high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
    low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
    _, ax = mpl.pyplot.subplots()
    high.plot(ax=ax)
    low.plot(ax=ax)
    for line in ax.get_lines():
        assert PeriodIndex(data=line.get_xdata()).freq == idxh.freq