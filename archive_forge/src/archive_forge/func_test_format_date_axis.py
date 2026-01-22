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
@pytest.mark.xfail(reason='Api changed in 3.6.0')
def test_format_date_axis(self):
    rng = date_range('1/1/2012', periods=12, freq='ME')
    df = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 3)), rng)
    _, ax = mpl.pyplot.subplots()
    ax = df.plot(ax=ax)
    xaxis = ax.get_xaxis()
    for line in xaxis.get_ticklabels():
        if len(line.get_text()) > 0:
            assert line.get_rotation() == 30