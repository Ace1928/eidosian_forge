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
def test_fontsize_set_correctly(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 9)), index=range(10))
    _, ax = mpl.pyplot.subplots()
    df.plot(fontsize=2, ax=ax)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        assert label.get_fontsize() == 2