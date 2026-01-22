from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('box', [date, datetime, Timestamp])
@pytest.mark.parametrize('side, expected', [('left', 4), ('right', 5)])
def test_get_slice_bounds_datetime_within(self, box, side, expected, tz_aware_fixture):
    tz = tz_aware_fixture
    index = bdate_range('2000-01-03', '2000-02-11').tz_localize(tz)
    key = box(year=2000, month=1, day=7)
    if tz is not None:
        with pytest.raises(TypeError, match='Cannot compare tz-naive'):
            index.get_slice_bound(key, side=side)
    else:
        result = index.get_slice_bound(key, side=side)
        assert result == expected