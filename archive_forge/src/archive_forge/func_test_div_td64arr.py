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
@pytest.mark.parametrize('box_cls', [np.array, Index, Series])
@pytest.mark.parametrize('left', lefts, ids=lambda x: type(x).__name__ + str(x.dtype))
def test_div_td64arr(self, left, box_cls):
    right = np.array([10, 40, 90], dtype='m8[s]')
    right = box_cls(right)
    expected = TimedeltaIndex(['1s', '2s', '3s'], dtype=right.dtype)
    if isinstance(left, Series) or box_cls is Series:
        expected = Series(expected)
    assert expected.dtype == right.dtype
    result = right / left
    tm.assert_equal(result, expected)
    result = right // left
    tm.assert_equal(result, expected)
    msg = "ufunc '(true_)?divide' cannot use operands with types"
    with pytest.raises(TypeError, match=msg):
        left / right
    msg = "ufunc 'floor_divide' cannot use operands with types"
    with pytest.raises(TypeError, match=msg):
        left // right