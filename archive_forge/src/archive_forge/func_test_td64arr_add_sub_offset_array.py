from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_add_sub_offset_array(self, box_with_array):
    box = box_with_array
    tdi = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'])
    other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])
    expected = TimedeltaIndex([tdi[n] + other[n] for n in range(len(tdi))], freq='infer')
    expected_sub = TimedeltaIndex([tdi[n] - other[n] for n in range(len(tdi))], freq='infer')
    tdi = tm.box_expected(tdi, box)
    expected = tm.box_expected(expected, box).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        res = tdi + other
    tm.assert_equal(res, expected)
    with tm.assert_produces_warning(PerformanceWarning):
        res2 = other + tdi
    tm.assert_equal(res2, expected)
    expected_sub = tm.box_expected(expected_sub, box_with_array).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        res_sub = tdi - other
    tm.assert_equal(res_sub, expected_sub)