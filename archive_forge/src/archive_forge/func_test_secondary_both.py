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
def test_secondary_both(self):
    ser = Series(np.random.default_rng(2).standard_normal(10))
    ser2 = Series(np.random.default_rng(2).standard_normal(10))
    ax = ser2.plot()
    ax2 = ser.plot(secondary_y=True)
    assert ax.get_yaxis().get_visible()
    assert not hasattr(ax, 'left_ax')
    assert hasattr(ax, 'right_ax')
    assert hasattr(ax2, 'left_ax')
    assert not hasattr(ax2, 'right_ax')