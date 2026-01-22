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
def test_time_change_xlim(self):
    t = datetime(1, 1, 1, 3, 30, 0)
    deltas = np.random.default_rng(2).integers(1, 20, 3).cumsum()
    ts = np.array([(t + timedelta(minutes=int(x))).time() for x in deltas])
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(len(ts)), 'b': np.random.default_rng(2).standard_normal(len(ts))}, index=ts)
    _, ax = mpl.pyplot.subplots()
    df.plot(ax=ax)
    ticks = ax.get_xticks()
    labels = ax.get_xticklabels()
    for _tick, _label in zip(ticks, labels):
        m, s = divmod(int(_tick), 60)
        h, m = divmod(m, 60)
        rs = _label.get_text()
        if len(rs) > 0:
            if s != 0:
                xp = time(h, m, s).strftime('%H:%M:%S')
            else:
                xp = time(h, m, s).strftime('%H:%M')
            assert xp == rs
    ax.set_xlim('1:30', '5:00')
    ticks = ax.get_xticks()
    labels = ax.get_xticklabels()
    for _tick, _label in zip(ticks, labels):
        m, s = divmod(int(_tick), 60)
        h, m = divmod(m, 60)
        rs = _label.get_text()
        if len(rs) > 0:
            if s != 0:
                xp = time(h, m, s).strftime('%H:%M:%S')
            else:
                xp = time(h, m, s).strftime('%H:%M')
            assert xp == rs