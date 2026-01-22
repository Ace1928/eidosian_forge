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
@pytest.mark.parametrize('op', [operator.truediv, operator.floordiv])
def test_div_negative_zero(self, zero, numeric_idx, op):
    if numeric_idx.dtype == np.uint64:
        pytest.skip(f'Div by negative 0 not relevant for {numeric_idx.dtype}')
    idx = numeric_idx - 3
    expected = Index([-np.inf, -np.inf, -np.inf, np.nan, np.inf], dtype=np.float64)
    expected = adjust_negative_zero(zero, expected)
    result = op(idx, zero)
    tm.assert_index_equal(result, expected)