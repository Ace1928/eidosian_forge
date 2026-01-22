import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
def test_compare_scalar_na(self, op, interval_array, nulls_fixture, box_with_array):
    box = box_with_array
    obj = tm.box_expected(interval_array, box)
    result = op(obj, nulls_fixture)
    if nulls_fixture is pd.NA:
        exp = np.ones(interval_array.shape, dtype=bool)
        expected = BooleanArray(exp, exp)
    else:
        expected = self.elementwise_comparison(op, interval_array, nulls_fixture)
    if not (box is Index and nulls_fixture is pd.NA):
        xbox = get_upcast_box(obj, nulls_fixture, True)
        expected = tm.box_expected(expected, xbox)
    tm.assert_equal(result, expected)
    rev = op(nulls_fixture, obj)
    tm.assert_equal(rev, expected)