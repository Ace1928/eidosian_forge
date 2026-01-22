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
def test_dt64arr_add_sub_td64_nat(self, box_with_array, tz_naive_fixture):
    tz = tz_naive_fixture
    dti = date_range('1994-04-01', periods=9, tz=tz, freq='QS')
    other = np.timedelta64('NaT')
    expected = DatetimeIndex(['NaT'] * 9, tz=tz).as_unit('ns')
    obj = tm.box_expected(dti, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = obj + other
    tm.assert_equal(result, expected)
    result = other + obj
    tm.assert_equal(result, expected)
    result = obj - other
    tm.assert_equal(result, expected)
    msg = 'cannot subtract'
    with pytest.raises(TypeError, match=msg):
        other - obj