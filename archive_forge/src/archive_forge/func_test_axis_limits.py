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
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
@pytest.mark.parametrize('obj', [Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)), DataFrame({'a': Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)), 'b': Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10)) + 1})])
def test_axis_limits(self, obj):
    _, ax = mpl.pyplot.subplots()
    obj.plot(ax=ax)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 5, xlim[1] + 10)
    result = ax.get_xlim()
    assert result[0] == xlim[0] - 5
    assert result[1] == xlim[1] + 10
    expected = (Period('1/1/2000', ax.freq), Period('4/1/2000', ax.freq))
    ax.set_xlim('1/1/2000', '4/1/2000')
    result = ax.get_xlim()
    assert int(result[0]) == expected[0].ordinal
    assert int(result[1]) == expected[1].ordinal
    expected = (Period('1/1/2000', ax.freq), Period('4/1/2000', ax.freq))
    ax.set_xlim(datetime(2000, 1, 1), datetime(2000, 4, 1))
    result = ax.get_xlim()
    assert int(result[0]) == expected[0].ordinal
    assert int(result[1]) == expected[1].ordinal
    fig = ax.get_figure()
    mpl.pyplot.close(fig)