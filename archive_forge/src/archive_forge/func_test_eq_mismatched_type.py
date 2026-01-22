import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
@pytest.mark.parametrize('other', ['a', pd.Timestamp(2017, 1, 1, 12), np.timedelta64(4)])
def test_eq_mismatched_type(self, other):
    arr = pd.array([True, False])
    result = arr == other
    expected = pd.array([False, False])
    tm.assert_extension_array_equal(result, expected)
    result = arr != other
    expected = pd.array([True, True])
    tm.assert_extension_array_equal(result, expected)