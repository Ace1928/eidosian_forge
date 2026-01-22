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
def test_fake_inferred_business(self):
    _, ax = mpl.pyplot.subplots()
    rng = date_range('2001-1-1', '2001-1-10')
    ts = Series(range(len(rng)), index=rng)
    ts = concat([ts[:3], ts[5:]])
    ts.plot(ax=ax)
    assert not hasattr(ax, 'freq')