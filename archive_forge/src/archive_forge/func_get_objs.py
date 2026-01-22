from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def get_objs():
    indexes = [Index([True, False] * 5, name='a'), Index(np.arange(10), dtype=np.int64, name='a'), Index(np.arange(10), dtype=np.float64, name='a'), DatetimeIndex(date_range('2020-01-01', periods=10), name='a'), DatetimeIndex(date_range('2020-01-01', periods=10), name='a').tz_localize(tz='US/Eastern'), PeriodIndex(period_range('2020-01-01', periods=10, freq='D'), name='a'), Index([str(i) for i in range(10)], name='a')]
    arr = np.random.default_rng(2).standard_normal(10)
    series = [Series(arr, index=idx, name='a') for idx in indexes]
    objs = indexes + series
    return objs