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
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_ts_plot_with_tz(self, tz_aware_fixture):
    tz = tz_aware_fixture
    index = date_range('1/1/2011', periods=2, freq='h', tz=tz)
    ts = Series([188.5, 328.25], index=index)
    _check_plot_works(ts.plot)
    ax = ts.plot()
    xdata = next(iter(ax.get_lines())).get_xdata()
    assert (xdata[0].hour, xdata[0].minute) == (0, 0)
    assert (xdata[-1].hour, xdata[-1].minute) == (1, 0)