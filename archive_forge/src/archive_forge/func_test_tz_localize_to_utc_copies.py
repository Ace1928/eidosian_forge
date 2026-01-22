from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
def test_tz_localize_to_utc_copies():
    arr = np.arange(5, dtype='i8')
    result = tz_convert_from_utc(arr, tz=UTC)
    tm.assert_numpy_array_equal(result, arr)
    assert not np.shares_memory(arr, result)
    result = tz_convert_from_utc(arr, tz=None)
    tm.assert_numpy_array_equal(result, arr)
    assert not np.shares_memory(arr, result)