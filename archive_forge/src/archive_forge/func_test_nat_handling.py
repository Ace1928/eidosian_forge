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
def test_nat_handling(self):
    _, ax = mpl.pyplot.subplots()
    dti = DatetimeIndex(['2015-01-01', NaT, '2015-01-03'])
    s = Series(range(len(dti)), dti)
    s.plot(ax=ax)
    xdata = ax.get_lines()[0].get_xdata()
    assert s.index.min() <= Series(xdata).min()
    assert Series(xdata).max() <= s.index.max()