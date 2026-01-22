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
@pytest.mark.slow
def test_finder_minutely(self):
    nminutes = 50 * 24 * 60
    rng = date_range('1/1/1999', freq='Min', periods=nminutes)
    ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    _, ax = mpl.pyplot.subplots()
    ser.plot(ax=ax)
    xaxis = ax.get_xaxis()
    rs = xaxis.get_majorticklocs()[0]
    xp = Period('1/1/1999', freq='Min').ordinal
    assert rs == xp