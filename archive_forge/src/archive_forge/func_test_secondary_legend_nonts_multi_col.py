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
def test_secondary_legend_nonts_multi_col(self):
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    fig = mpl.pyplot.figure()
    ax = fig.add_subplot(211)
    ax = df.plot(secondary_y=['C', 'D'], ax=ax)
    leg = ax.get_legend()
    assert len(leg.get_lines()) == 4
    assert ax.right_ax.get_legend() is None
    colors = set()
    for line in leg.get_lines():
        colors.add(line.get_color())
    assert len(colors) == 4