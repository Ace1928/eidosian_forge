import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('freq', ['M', '2M', '3M'])
def test_parr_cmp_pi(self, freq, box_with_array):
    base = PeriodIndex(['2011-01', '2011-02', '2011-03', '2011-04'], freq=freq)
    base = tm.box_expected(base, box_with_array)
    idx = PeriodIndex(['2011-02', '2011-01', '2011-03', '2011-05'], freq=freq)
    xbox = get_upcast_box(base, idx, True)
    exp = np.array([False, False, True, False])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base == idx, exp)
    exp = np.array([True, True, False, True])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base != idx, exp)
    exp = np.array([False, True, False, False])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base > idx, exp)
    exp = np.array([True, False, False, True])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base < idx, exp)
    exp = np.array([False, True, True, False])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base >= idx, exp)
    exp = np.array([True, False, True, True])
    exp = tm.box_expected(exp, xbox)
    tm.assert_equal(base <= idx, exp)