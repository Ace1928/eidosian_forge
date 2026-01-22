import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
@pytest.mark.parametrize('other, expected', [(pd.NA, [None, False, None]), (True, [True, False, None]), (False, [False, False, False]), (np.bool_(True), [True, False, None]), (np.bool_(False), [False, False, False])])
def test_kleene_and_scalar(self, other, expected):
    a = pd.array([True, False, None], dtype='boolean')
    result = a & other
    expected = pd.array(expected, dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = other & a
    tm.assert_extension_array_equal(result, expected)
    tm.assert_extension_array_equal(a, pd.array([True, False, None], dtype='boolean'))