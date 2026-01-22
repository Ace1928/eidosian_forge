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
def test_mixed_freq_regular_first_df(self):
    s1 = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20, freq='B')).to_frame()
    s2 = s1.iloc[[0, 5, 10, 11, 12, 13, 14, 15], :]
    _, ax = mpl.pyplot.subplots()
    s1.plot(ax=ax)
    ax2 = s2.plot(style='g', ax=ax)
    lines = ax2.get_lines()
    msg = 'PeriodDtype\\[B\\] is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        idx1 = PeriodIndex(lines[0].get_xdata())
        idx2 = PeriodIndex(lines[1].get_xdata())
        assert idx1.equals(s1.index.to_period('B'))
        assert idx2.equals(s2.index.to_period('B'))
        left, right = ax2.get_xlim()
        pidx = s1.index.to_period()
    assert left <= pidx[0].ordinal
    assert right >= pidx[-1].ordinal