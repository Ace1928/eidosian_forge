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
@pytest.mark.parametrize('data', [[Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [Timedelta('1 days'), NaT, Timedelta('3 days')], [Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')]])
@pytest.mark.parametrize('dtype', [None, object])
def test_nat_comparisons_scalar(self, dtype, data, box_with_array):
    box = box_with_array
    left = Series(data, dtype=dtype)
    left = tm.box_expected(left, box)
    xbox = get_upcast_box(left, NaT, True)
    expected = [False, False, False]
    expected = tm.box_expected(expected, xbox)
    if box is pd.array and dtype is object:
        expected = pd.array(expected, dtype='bool')
    tm.assert_equal(left == NaT, expected)
    tm.assert_equal(NaT == left, expected)
    expected = [True, True, True]
    expected = tm.box_expected(expected, xbox)
    if box is pd.array and dtype is object:
        expected = pd.array(expected, dtype='bool')
    tm.assert_equal(left != NaT, expected)
    tm.assert_equal(NaT != left, expected)
    expected = [False, False, False]
    expected = tm.box_expected(expected, xbox)
    if box is pd.array and dtype is object:
        expected = pd.array(expected, dtype='bool')
    tm.assert_equal(left < NaT, expected)
    tm.assert_equal(NaT > left, expected)
    tm.assert_equal(left <= NaT, expected)
    tm.assert_equal(NaT >= left, expected)
    tm.assert_equal(left > NaT, expected)
    tm.assert_equal(NaT < left, expected)
    tm.assert_equal(left >= NaT, expected)
    tm.assert_equal(NaT <= left, expected)