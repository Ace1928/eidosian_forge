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
@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.floordiv, operator.truediv, operator.pow, operator.mod])
def test_arithmetic_with_frame_or_series(self, op):
    index = RangeIndex(5)
    other = Series(np.random.default_rng(2).standard_normal(5))
    expected = op(Series(index), other)
    result = op(index, other)
    tm.assert_series_equal(result, expected)
    other = pd.DataFrame(np.random.default_rng(2).standard_normal((2, 5)))
    expected = op(pd.DataFrame([index, index]), other)
    result = op(index, other)
    tm.assert_frame_equal(result, expected)