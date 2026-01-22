from datetime import (
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('dt_string, expected_tz', [['01-01-2013 08:00:00+08:00', 480], ['2013-01-01T08:00:00.000000000+0800', 480], ['2012-12-31T16:00:00.000000000-0800', -480], ['12-31-2012 23:00:00-01:00', -60]])
def test_parsing_timezone_offsets(dt_string, expected_tz):
    arr = np.array(['01-01-2013 00:00:00'], dtype=object)
    expected, _ = tslib.array_to_datetime(arr)
    arr = np.array([dt_string], dtype=object)
    result, result_tz = tslib.array_to_datetime(arr)
    tm.assert_numpy_array_equal(result, expected)
    assert result_tz == timezone(timedelta(minutes=expected_tz))