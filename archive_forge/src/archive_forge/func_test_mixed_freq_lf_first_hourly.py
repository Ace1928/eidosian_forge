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
def test_mixed_freq_lf_first_hourly(self):
    idxh = date_range('1/1/1999', periods=240, freq='min')
    idxl = date_range('1/1/1999', periods=4, freq='h')
    high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
    low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
    _, ax = mpl.pyplot.subplots()
    low.plot(ax=ax)
    high.plot(ax=ax)
    for line in ax.get_lines():
        assert PeriodIndex(data=line.get_xdata()).freq == 'min'