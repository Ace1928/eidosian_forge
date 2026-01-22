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
def test_plot_outofbounds_datetime(self):
    values = [date(1677, 1, 1), date(1677, 1, 2)]
    _, ax = mpl.pyplot.subplots()
    ax.plot(values)
    values = [datetime(1677, 1, 1, 12), datetime(1677, 1, 2, 12)]
    ax.plot(values)