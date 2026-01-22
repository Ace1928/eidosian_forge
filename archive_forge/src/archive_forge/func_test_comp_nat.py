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
@pytest.mark.parametrize('dtype', [object, None])
def test_comp_nat(self, dtype):
    left = PeriodIndex([Period('2011-01-01'), pd.NaT, Period('2011-01-03')])
    right = PeriodIndex([pd.NaT, pd.NaT, Period('2011-01-03')])
    if dtype is not None:
        left = left.astype(dtype)
        right = right.astype(dtype)
    result = left == right
    expected = np.array([False, False, True])
    tm.assert_numpy_array_equal(result, expected)
    result = left != right
    expected = np.array([True, True, False])
    tm.assert_numpy_array_equal(result, expected)
    expected = np.array([False, False, False])
    tm.assert_numpy_array_equal(left == pd.NaT, expected)
    tm.assert_numpy_array_equal(pd.NaT == right, expected)
    expected = np.array([True, True, True])
    tm.assert_numpy_array_equal(left != pd.NaT, expected)
    tm.assert_numpy_array_equal(pd.NaT != left, expected)
    expected = np.array([False, False, False])
    tm.assert_numpy_array_equal(left < pd.NaT, expected)
    tm.assert_numpy_array_equal(pd.NaT > left, expected)