from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('scalar_td', [Timedelta(days=1), Timedelta(days=1).to_timedelta64(), Timedelta(days=1).to_pytimedelta()], ids=lambda x: type(x).__name__)
@pytest.mark.parametrize('dtype', [np.int64, np.float64])
def test_numeric_arr_mul_tdscalar_numexpr_path(self, dtype, scalar_td, box_with_array):
    box = box_with_array
    arr_i8 = np.arange(2 * 10 ** 4).astype(np.int64, copy=False)
    arr = arr_i8.astype(dtype, copy=False)
    obj = tm.box_expected(arr, box, transpose=False)
    expected = arr_i8.view('timedelta64[D]').astype('timedelta64[ns]')
    if type(scalar_td) is timedelta:
        expected = expected.astype('timedelta64[us]')
    expected = tm.box_expected(expected, box, transpose=False)
    result = obj * scalar_td
    tm.assert_equal(result, expected)
    result = scalar_td * obj
    tm.assert_equal(result, expected)