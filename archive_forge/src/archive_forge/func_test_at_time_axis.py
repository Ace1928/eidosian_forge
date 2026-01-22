from datetime import time
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('axis', ['index', 'columns', 0, 1])
def test_at_time_axis(self, axis):
    rng = date_range('1/1/2000', '1/5/2000', freq='5min')
    ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), len(rng))))
    ts.index, ts.columns = (rng, rng)
    indices = rng[(rng.hour == 9) & (rng.minute == 30) & (rng.second == 0)]
    if axis in ['index', 0]:
        expected = ts.loc[indices, :]
    elif axis in ['columns', 1]:
        expected = ts.loc[:, indices]
    result = ts.at_time('9:30', axis=axis)
    result.index = result.index._with_freq(None)
    expected.index = expected.index._with_freq(None)
    tm.assert_frame_equal(result, expected)