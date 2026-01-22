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
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_mixed_freq_irreg_period(self):
    ts = Series(np.arange(30, dtype=np.float64), index=date_range('2020-01-01', periods=30))
    irreg = ts.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 29]]
    msg = 'PeriodDtype\\[B\\] is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rng = period_range('1/3/2000', periods=30, freq='B')
    ps = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
    _, ax = mpl.pyplot.subplots()
    irreg.plot(ax=ax)
    ps.plot(ax=ax)