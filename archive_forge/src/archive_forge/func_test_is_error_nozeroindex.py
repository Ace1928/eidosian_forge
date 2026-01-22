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
def test_is_error_nozeroindex(self):
    i = np.array([1, 2, 3])
    a = DataFrame(i, index=i)
    _check_plot_works(a.plot, xerr=a)
    _check_plot_works(a.plot, yerr=a)