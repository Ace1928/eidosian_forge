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
def test_irreg_hf(self):
    idx = date_range('2012-6-22 21:59:51', freq='s', periods=10)
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx)
    irreg = df.iloc[[0, 1, 3, 4]]
    _, ax = mpl.pyplot.subplots()
    irreg.plot(ax=ax)
    diffs = Series(ax.get_lines()[0].get_xydata()[:, 0]).diff()
    sec = 1.0 / 24 / 60 / 60
    assert (np.fabs(diffs[1:] - [sec, sec * 2, sec]) < 1e-08).all()