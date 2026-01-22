import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
@pytest.mark.parametrize('other', [True, False, pd.NA, [True, False, None] * 3])
def test_no_masked_assumptions(self, other, all_logical_operators):
    a = pd.arrays.BooleanArray(np.array([True, True, True, False, False, False, True, False, True]), np.array([False] * 6 + [True, True, True]))
    b = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype='boolean')
    if isinstance(other, list):
        other = pd.array(other, dtype='boolean')
    result = getattr(a, all_logical_operators)(other)
    expected = getattr(b, all_logical_operators)(other)
    tm.assert_extension_array_equal(result, expected)
    if isinstance(other, BooleanArray):
        other._data[other._mask] = True
        a._data[a._mask] = False
        result = getattr(a, all_logical_operators)(other)
        expected = getattr(b, all_logical_operators)(other)
        tm.assert_extension_array_equal(result, expected)