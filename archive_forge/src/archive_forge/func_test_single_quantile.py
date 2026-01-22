import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('data,start,end', [(9.0, 8.999, 9.0), (0.0, -0.001, 0.0), (-9.0, -9.001, -9.0)])
@pytest.mark.parametrize('length', [1, 2])
@pytest.mark.parametrize('labels', [None, False])
def test_single_quantile(data, start, end, length, labels):
    ser = Series([data] * length)
    result = qcut(ser, 1, labels=labels)
    if labels is None:
        intervals = IntervalIndex([Interval(start, end)] * length, closed='right')
        expected = Series(intervals).astype(CategoricalDtype(ordered=True))
    else:
        expected = Series([0] * length, dtype=np.intp)
    tm.assert_series_equal(result, expected)