from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_calendar_fourier(reset_randomstate):
    inc = np.abs(np.random.standard_normal(1000))
    inc = np.cumsum(inc)
    inc = 10 * inc / inc[-1]
    offset = (24 * 3600 * inc).astype(np.int64)
    base = pd.Timestamp('2000-1-1')
    index = [base + pd.Timedelta(val, unit='s') for val in offset]
    index = pd.Index(index)
    cf = CalendarFourier('D', 2)
    assert cf.order == 2
    terms = cf.in_sample(index)
    cols = []
    for i in range(2 * cf.order):
        fn = 'cos' if i % 2 else 'sin'
        cols.append(f'{fn}({i // 2 + 1},freq=D)')
    assert list(terms.columns) == cols
    inc = offset / (24 * 3600)
    loc = 2 * np.pi * (inc - np.floor(inc))
    expected = []
    for i in range(4):
        scale = i // 2 + 1
        fn = np.cos if i % 2 else np.sin
        expected.append(fn(scale * loc))
    expected = np.column_stack(expected)
    np.testing.assert_allclose(expected, terms.values)