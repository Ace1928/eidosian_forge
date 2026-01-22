from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
def test_datetime64_with_index(self):
    ser = Series(np.random.default_rng(2).standard_normal(5))
    expected = ser - ser.index.to_series()
    result = ser - ser.index
    tm.assert_series_equal(result, expected)
    ser = Series(date_range('20130101', periods=5), index=date_range('20130101', periods=5))
    expected = ser - ser.index.to_series()
    result = ser - ser.index
    tm.assert_series_equal(result, expected)
    msg = 'cannot subtract PeriodArray from DatetimeArray'
    with pytest.raises(TypeError, match=msg):
        result = ser - ser.index.to_period()
    df = pd.DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=date_range('20130101', periods=5))
    df['date'] = pd.Timestamp('20130102')
    df['expected'] = df['date'] - df.index.to_series()
    df['result'] = df['date'] - df.index
    tm.assert_series_equal(df['result'], df['expected'], check_names=False)