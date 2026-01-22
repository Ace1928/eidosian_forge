import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
def test_kleene_xor(self):
    a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype='boolean')
    b = pd.array([True, False, None] * 3, dtype='boolean')
    result = a ^ b
    expected = pd.array([False, True, None, True, False, None, None, None, None], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = b ^ a
    tm.assert_extension_array_equal(result, expected)
    tm.assert_extension_array_equal(a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype='boolean'))
    tm.assert_extension_array_equal(b, pd.array([True, False, None] * 3, dtype='boolean'))