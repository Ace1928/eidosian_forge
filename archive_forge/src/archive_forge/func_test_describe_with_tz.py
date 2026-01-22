import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_describe_with_tz(self, tz_naive_fixture):
    tz = tz_naive_fixture
    name = str(tz_naive_fixture)
    start = Timestamp(2018, 1, 1)
    end = Timestamp(2018, 1, 5)
    s = Series(date_range(start, end, tz=tz), name=name)
    result = s.describe()
    expected = Series([5, Timestamp(2018, 1, 3).tz_localize(tz), start.tz_localize(tz), s[1], s[2], s[3], end.tz_localize(tz)], name=name, index=['count', 'mean', 'min', '25%', '50%', '75%', 'max'])
    tm.assert_series_equal(result, expected)