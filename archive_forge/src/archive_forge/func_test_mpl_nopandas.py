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
def test_mpl_nopandas(self):
    dates = [date(2008, 12, 31), date(2009, 1, 31)]
    values1 = np.arange(10.0, 11.0, 0.5)
    values2 = np.arange(11.0, 12.0, 0.5)
    kw = {'fmt': '-', 'lw': 4}
    _, ax = mpl.pyplot.subplots()
    ax.plot_date([x.toordinal() for x in dates], values1, **kw)
    ax.plot_date([x.toordinal() for x in dates], values2, **kw)
    line1, line2 = ax.get_lines()
    exp = np.array([x.toordinal() for x in dates], dtype=np.float64)
    tm.assert_numpy_array_equal(line1.get_xydata()[:, 0], exp)
    exp = np.array([x.toordinal() for x in dates], dtype=np.float64)
    tm.assert_numpy_array_equal(line2.get_xydata()[:, 0], exp)