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
def test_format_timedelta_ticks_wide(self):
    expected_labels = ['00:00:00', '1 days 03:46:40', '2 days 07:33:20', '3 days 11:20:00', '4 days 15:06:40', '5 days 18:53:20', '6 days 22:40:00', '8 days 02:26:40', '9 days 06:13:20']
    rng = timedelta_range('0', periods=10, freq='1 d')
    df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
    _, ax = mpl.pyplot.subplots()
    ax = df.plot(fontsize=2, ax=ax)
    mpl.pyplot.draw()
    labels = ax.get_xticklabels()
    result_labels = [x.get_text() for x in labels]
    assert len(result_labels) == len(expected_labels)
    assert result_labels == expected_labels