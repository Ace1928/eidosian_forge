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
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_dt64arr_series_add_tick_DateOffset(self, box_with_array, unit):
    ser = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')]).dt.as_unit(unit)
    expected = Series([Timestamp('20130101 9:01:05'), Timestamp('20130101 9:02:05')]).dt.as_unit(unit)
    ser = tm.box_expected(ser, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = ser + pd.offsets.Second(5)
    tm.assert_equal(result, expected)
    result2 = pd.offsets.Second(5) + ser
    tm.assert_equal(result2, expected)