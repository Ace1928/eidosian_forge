from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
def test_resample_offset_with_timedeltaindex():
    rng = timedelta_range(start='0s', periods=25, freq='s')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    with_base = ts.resample('2s', offset='5s').mean()
    without_base = ts.resample('2s').mean()
    exp_without_base = timedelta_range(start='0s', end='25s', freq='2s')
    exp_with_base = timedelta_range(start='5s', end='29s', freq='2s')
    tm.assert_index_equal(without_base.index, exp_without_base)
    tm.assert_index_equal(with_base.index, exp_with_base)