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
def test_finder_monthly_long(self):
    rng = period_range('1988Q1', periods=24 * 12, freq='M')
    ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    _, ax = mpl.pyplot.subplots()
    ser.plot(ax=ax)
    xaxis = ax.get_xaxis()
    rs = xaxis.get_majorticklocs()[0]
    xp = Period('1989Q1', 'M').ordinal
    assert rs == xp