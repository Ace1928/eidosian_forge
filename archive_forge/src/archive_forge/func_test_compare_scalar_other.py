import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
@pytest.mark.parametrize('other', [0, 1.0, True, 'foo', Timestamp('2017-01-01'), Timestamp('2017-01-01', tz='US/Eastern'), Timedelta('0 days'), Period('2017-01-01', 'D')])
def test_compare_scalar_other(self, op, interval_array, other):
    result = op(interval_array, other)
    expected = self.elementwise_comparison(op, interval_array, other)
    tm.assert_numpy_array_equal(result, expected)