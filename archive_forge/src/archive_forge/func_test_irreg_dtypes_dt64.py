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
def test_irreg_dtypes_dt64(self):
    idx = date_range('1/1/2000', periods=10)
    idx = idx[[0, 2, 5, 9]].astype(object)
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 3)), idx)
    _, ax = mpl.pyplot.subplots()
    _check_plot_works(df.plot, ax=ax)