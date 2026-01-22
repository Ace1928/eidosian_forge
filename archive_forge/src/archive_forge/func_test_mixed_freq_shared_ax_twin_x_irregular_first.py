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
@pytest.mark.xfail(reason='TODO (GH14330, GH14322)')
def test_mixed_freq_shared_ax_twin_x_irregular_first(self):
    idx1 = date_range('2015-01-01', periods=3, freq='M')
    idx2 = idx1[:1].union(idx1[2:])
    s1 = Series(range(len(idx1)), idx1)
    s2 = Series(range(len(idx2)), idx2)
    _, ax1 = mpl.pyplot.subplots()
    ax2 = ax1.twinx()
    s2.plot(ax=ax1)
    s1.plot(ax=ax2)
    assert ax1.lines[0].get_xydata()[0, 0] == ax2.lines[0].get_xydata()[0, 0]