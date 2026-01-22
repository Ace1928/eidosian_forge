from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arr', [pytest.param(np.array([], dtype=np.int64), id='empty'), pytest.param(np.array([iNaT], dtype=np.int64), id='all_nat')])
def test_tz_convert_corner(arr):
    result = tz_convert_from_utc(arr, timezones.maybe_get_tz('Asia/Tokyo'))
    tm.assert_numpy_array_equal(result, arr)