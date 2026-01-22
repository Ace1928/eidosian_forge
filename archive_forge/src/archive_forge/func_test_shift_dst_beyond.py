import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ex', [10, -10, 20, -20])
def test_shift_dst_beyond(self, frame_or_series, ex):
    dates = date_range('2016-11-06', freq='h', periods=10, tz='US/Eastern')
    obj = frame_or_series(dates)
    res = obj.shift(ex)
    exp = frame_or_series([NaT] * 10, dtype='datetime64[ns, US/Eastern]')
    tm.assert_equal(res, exp)
    assert tm.get_dtype(res) == 'datetime64[ns, US/Eastern]'