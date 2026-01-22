import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('func,np_func,expected,np_kwargs', [('count', len, [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, np.nan], {}), ('min', np.min, [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0, np.nan], {}), ('max', np.max, [2.0, 3.0, 4.0, 100.0, 100.0, 100.0, 8.0, 9.0, 9.0, np.nan], {}), ('std', np.std, [1.0, 1.0, 1.0, 55.71654452, 54.85739087, 53.9845657, 1.0, 1.0, 0.70710678, np.nan], {'ddof': 1}), ('var', np.var, [1.0, 1.0, 1.0, 3104.333333, 3009.333333, 2914.333333, 1.0, 1.0, 0.5, np.nan], {'ddof': 1}), ('median', np.median, [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 8.5, np.nan], {})])
def test_rolling_forward_window(frame_or_series, func, np_func, expected, np_kwargs, step):
    values = np.arange(10.0)
    values[5] = 100.0
    indexer = FixedForwardWindowIndexer(window_size=3)
    match = "Forward-looking windows can't have center=True"
    with pytest.raises(ValueError, match=match):
        rolling = frame_or_series(values).rolling(window=indexer, center=True)
        getattr(rolling, func)()
    match = "Forward-looking windows don't support setting the closed argument"
    with pytest.raises(ValueError, match=match):
        rolling = frame_or_series(values).rolling(window=indexer, closed='right')
        getattr(rolling, func)()
    rolling = frame_or_series(values).rolling(window=indexer, min_periods=2, step=step)
    result = getattr(rolling, func)()
    expected = frame_or_series(expected)[::step]
    tm.assert_equal(result, expected)
    expected2 = frame_or_series(rolling.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result, expected2)
    min_periods = 0 if func == 'count' else None
    rolling3 = frame_or_series(values).rolling(window=indexer, min_periods=min_periods)
    result3 = getattr(rolling3, func)()
    expected3 = frame_or_series(rolling3.apply(lambda x: np_func(x, **np_kwargs)))
    tm.assert_equal(result3, expected3)