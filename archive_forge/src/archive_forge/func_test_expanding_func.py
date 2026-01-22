import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func,static_comp', [('sum', np.sum), ('mean', lambda x: np.mean(x, axis=0)), ('max', lambda x: np.max(x, axis=0)), ('min', lambda x: np.min(x, axis=0))], ids=['sum', 'mean', 'max', 'min'])
def test_expanding_func(func, static_comp, frame_or_series):
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    msg = "The 'axis' keyword in (Series|DataFrame).expanding is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        obj = data.expanding(min_periods=1, axis=0)
    result = getattr(obj, func)()
    assert isinstance(result, frame_or_series)
    msg = 'The behavior of DataFrame.sum with axis=None is deprecated'
    warn = None
    if frame_or_series is DataFrame and static_comp is np.sum:
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):
        expected = static_comp(data[:11])
    if frame_or_series is Series:
        tm.assert_almost_equal(result[10], expected)
    else:
        tm.assert_series_equal(result.iloc[10], expected, check_names=False)