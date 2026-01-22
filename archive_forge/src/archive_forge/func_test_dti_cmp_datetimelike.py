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
@pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
def test_dti_cmp_datetimelike(self, other, tz_naive_fixture):
    tz = tz_naive_fixture
    dti = date_range('2016-01-01', periods=2, tz=tz)
    if tz is not None:
        if isinstance(other, np.datetime64):
            pytest.skip(f'{type(other).__name__} is not tz aware')
        other = localize_pydatetime(other, dti.tzinfo)
    result = dti == other
    expected = np.array([True, False])
    tm.assert_numpy_array_equal(result, expected)
    result = dti > other
    expected = np.array([False, True])
    tm.assert_numpy_array_equal(result, expected)
    result = dti >= other
    expected = np.array([True, True])
    tm.assert_numpy_array_equal(result, expected)
    result = dti < other
    expected = np.array([False, False])
    tm.assert_numpy_array_equal(result, expected)
    result = dti <= other
    expected = np.array([True, False])
    tm.assert_numpy_array_equal(result, expected)