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
def test_dt64arr_sub_NaT(self, box_with_array, unit):
    dti = DatetimeIndex([NaT, Timestamp('19900315')]).as_unit(unit)
    ser = tm.box_expected(dti, box_with_array)
    result = ser - NaT
    expected = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)
    dti_tz = dti.tz_localize('Asia/Tokyo')
    ser_tz = tm.box_expected(dti_tz, box_with_array)
    result = ser_tz - NaT
    expected = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)