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
def test_dt64arr_timestamp_equality(self, box_with_array):
    box = box_with_array
    ser = Series([Timestamp('2000-01-29 01:59:00'), Timestamp('2000-01-30'), NaT])
    ser = tm.box_expected(ser, box)
    xbox = get_upcast_box(ser, ser, True)
    result = ser != ser
    expected = tm.box_expected([False, False, True], xbox)
    tm.assert_equal(result, expected)
    if box is pd.DataFrame:
        with pytest.raises(ValueError, match='not aligned'):
            ser != ser[0]
    else:
        result = ser != ser[0]
        expected = tm.box_expected([False, True, True], xbox)
        tm.assert_equal(result, expected)
    if box is pd.DataFrame:
        with pytest.raises(ValueError, match='not aligned'):
            ser != ser[2]
    else:
        result = ser != ser[2]
        expected = tm.box_expected([True, True, True], xbox)
        tm.assert_equal(result, expected)
    result = ser == ser
    expected = tm.box_expected([True, True, False], xbox)
    tm.assert_equal(result, expected)
    if box is pd.DataFrame:
        with pytest.raises(ValueError, match='not aligned'):
            ser == ser[0]
    else:
        result = ser == ser[0]
        expected = tm.box_expected([True, False, False], xbox)
        tm.assert_equal(result, expected)
    if box is pd.DataFrame:
        with pytest.raises(ValueError, match='not aligned'):
            ser == ser[2]
    else:
        result = ser == ser[2]
        expected = tm.box_expected([False, False, False], xbox)
        tm.assert_equal(result, expected)