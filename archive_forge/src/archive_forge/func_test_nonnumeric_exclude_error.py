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
def test_nonnumeric_exclude_error(self):
    idx = date_range('1/1/1987', freq='YE', periods=3)
    df = DataFrame({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]}, idx)
    msg = 'no numeric data to plot'
    with pytest.raises(TypeError, match=msg):
        df['A'].plot()