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
@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('delta', [1, 0, -1])
def test_addsub_arithmetic(self, dtype, delta):
    delta = dtype(delta)
    index = Index([10, 11, 12], dtype=dtype)
    result = index + delta
    expected = Index(index.values + delta, dtype=dtype)
    tm.assert_index_equal(result, expected)
    result = index - delta
    expected = Index(index.values - delta, dtype=dtype)
    tm.assert_index_equal(result, expected)
    tm.assert_index_equal(index + index, 2 * index)
    tm.assert_index_equal(index - index, 0 * index)
    assert not (index - index).empty