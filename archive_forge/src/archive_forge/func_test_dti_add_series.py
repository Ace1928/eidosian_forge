from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_dti_add_series(self, tz_naive_fixture, names):
    tz = tz_naive_fixture
    index = DatetimeIndex(['2016-06-28 05:30', '2016-06-28 05:31'], tz=tz, name=names[0]).as_unit('ns')
    ser = Series([Timedelta(seconds=5)] * 2, index=index, name=names[1])
    expected = Series(index + Timedelta(seconds=5), index=index, name=names[2])
    expected.name = names[2]
    assert expected.dtype == index.dtype
    result = ser + index
    tm.assert_series_equal(result, expected)
    result2 = index + ser
    tm.assert_series_equal(result2, expected)
    expected = index + Timedelta(seconds=5)
    result3 = ser.values + index
    tm.assert_index_equal(result3, expected)
    result4 = index + ser.values
    tm.assert_index_equal(result4, expected)