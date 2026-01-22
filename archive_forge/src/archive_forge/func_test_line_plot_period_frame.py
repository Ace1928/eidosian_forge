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
@pytest.mark.parametrize('freq', ['s', 'min', 'h', 'D', 'W', 'ME', 'QE', 'YE'])
def test_line_plot_period_frame(self, freq):
    idx = date_range('12/31/1999', freq=freq, periods=100)
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), index=idx, columns=['A', 'B', 'C'])
    _check_plot_works(df.plot, df.index.freq)