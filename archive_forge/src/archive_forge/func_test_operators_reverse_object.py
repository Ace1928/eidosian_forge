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
@pytest.mark.parametrize('op', [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv])
def test_operators_reverse_object(self, op):
    arr = Series(np.random.default_rng(2).standard_normal(10), index=np.arange(10), dtype=object)
    result = op(1.0, arr)
    expected = op(1.0, arr.astype(float))
    tm.assert_series_equal(result.astype(float), expected)