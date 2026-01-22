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
def test_secondary_upsample(self):
    idxh = date_range('1/1/1999', periods=365, freq='D')
    idxl = date_range('1/1/1999', periods=12, freq='ME')
    high = Series(np.random.default_rng(2).standard_normal(len(idxh)), idxh)
    low = Series(np.random.default_rng(2).standard_normal(len(idxl)), idxl)
    _, ax = mpl.pyplot.subplots()
    low.plot(ax=ax)
    ax = high.plot(secondary_y=True, ax=ax)
    for line in ax.get_lines():
        assert PeriodIndex(line.get_xdata()).freq == 'D'
    assert hasattr(ax, 'left_ax')
    assert not hasattr(ax, 'right_ax')
    for line in ax.left_ax.get_lines():
        assert PeriodIndex(line.get_xdata()).freq == 'D'