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
@pytest.mark.parametrize('transpose', [True, False])
def test_parr_add_sub_td64_nat(self, box_with_array, transpose):
    pi = period_range('1994-04-01', periods=9, freq='19D')
    other = np.timedelta64('NaT')
    expected = PeriodIndex(['NaT'] * 9, freq='19D')
    obj = tm.box_expected(pi, box_with_array, transpose=transpose)
    expected = tm.box_expected(expected, box_with_array, transpose=transpose)
    result = obj + other
    tm.assert_equal(result, expected)
    result = other + obj
    tm.assert_equal(result, expected)
    result = obj - other
    tm.assert_equal(result, expected)
    msg = 'cannot subtract .* from .*'
    with pytest.raises(TypeError, match=msg):
        other - obj