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
def test_mixed_freq_second_millisecond_low_to_high(self):
    idxh = date_range('2014-07-01 09:00', freq='s', periods=50)
    idxl = date_range('2014-07-01 09:00', freq='100ms', periods=500)
    high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
    low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
    _, ax = mpl.pyplot.subplots()
    low.plot(ax=ax)
    high.plot(ax=ax)
    assert len(ax.get_lines()) == 2
    for line in ax.get_lines():
        assert PeriodIndex(data=line.get_xdata()).freq == 'ms'