from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_calendar_time_trend(reset_randomstate):
    inc = np.abs(np.random.standard_normal(1000))
    inc = np.cumsum(inc)
    inc = 10 * inc / inc[-1]
    offset = (24 * 3600 * inc).astype(np.int64)
    base = pd.Timestamp('2000-1-1')
    index = [base + pd.Timedelta(val, 's') for val in offset]
    index = pd.Index(index)
    ctt = CalendarTimeTrend('D', True, order=3, base_period=base)
    assert ctt.order == 3
    terms = ctt.in_sample(index)
    cols = ['const', 'trend', 'trend_squared', 'trend_cubed']
    assert list(terms.columns) == cols
    inc = 1 + offset / (24 * 3600)
    expected = []
    for i in range(4):
        expected.append(inc ** i)
    expected = np.column_stack(expected)
    np.testing.assert_allclose(expected, terms.values)
    ctt = CalendarTimeTrend('D', True, order=2, base_period=base)
    ctt2 = CalendarTimeTrend.from_string('D', trend='ctt', base_period=base)
    pd.testing.assert_frame_equal(ctt.in_sample(index), ctt2.in_sample(index))
    ct = CalendarTimeTrend('D', True, order=1, base_period=base)
    ct2 = CalendarTimeTrend.from_string('D', trend='ct', base_period=base)
    pd.testing.assert_frame_equal(ct.in_sample(index), ct2.in_sample(index))
    ctttt = CalendarTimeTrend('D', True, order=4, base_period=base)
    assert ctttt.order == 4
    terms = ctttt.in_sample(index)
    cols = ['const', 'trend', 'trend_squared', 'trend_cubed', 'trend**4']
    assert list(terms.columns) == cols