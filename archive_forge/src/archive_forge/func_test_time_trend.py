from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_time_trend(index):
    tt = TimeTrend(constant=True)
    const = tt.in_sample(index)
    assert const.shape == (index.shape[0], 1)
    assert np.all(const == 1)
    pd.testing.assert_index_equal(const.index, index)
    warn = None
    if is_int_index(index) and np.any(np.diff(index) != 1) or (type(index) is pd.Index and max(index) > 2 ** 63):
        warn = UserWarning
    with pytest_warns(warn):
        const_fcast = tt.out_of_sample(23, index)
    assert np.all(const_fcast == 1)
    tt = TimeTrend(constant=False)
    empty = tt.in_sample(index)
    assert empty.shape == (index.shape[0], 0)
    tt = TimeTrend(constant=False, order=2)
    t2 = tt.in_sample(index)
    assert t2.shape == (index.shape[0], 2)
    assert list(t2.columns) == ['trend', 'trend_squared']
    tt = TimeTrend(constant=True, order=2)
    final = tt.in_sample(index)
    expected = pd.concat([const, t2], axis=1)
    pd.testing.assert_frame_equal(final, expected)
    tt = TimeTrend(constant=True, order=2)
    short = tt.in_sample(index[:-50])
    with pytest_warns(warn):
        remainder = tt.out_of_sample(50, index[:-50])
    direct = tt.out_of_sample(steps=50, index=index[:-50], forecast_index=index[-50:])
    combined = pd.concat([short, remainder], axis=0)
    if isinstance(index, (pd.DatetimeIndex, pd.RangeIndex)):
        pd.testing.assert_frame_equal(combined, final)
    combined = pd.concat([short, direct], axis=0)
    pd.testing.assert_frame_equal(combined, final, check_index_type=False)