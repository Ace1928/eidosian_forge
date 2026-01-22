from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_between_time_axis_raises(self, axis):
    rng = date_range('1/1/2000', periods=100, freq='10min')
    mask = np.arange(0, len(rng))
    rand_data = np.random.default_rng(2).standard_normal((len(rng), len(rng)))
    ts = DataFrame(rand_data, index=rng, columns=rng)
    stime, etime = ('08:00:00', '09:00:00')
    msg = 'Index must be DatetimeIndex'
    if axis in ['columns', 1]:
        ts.index = mask
        with pytest.raises(TypeError, match=msg):
            ts.between_time(stime, etime)
        with pytest.raises(TypeError, match=msg):
            ts.between_time(stime, etime, axis=0)
    if axis in ['index', 0]:
        ts.columns = mask
        with pytest.raises(TypeError, match=msg):
            ts.between_time(stime, etime, axis=1)