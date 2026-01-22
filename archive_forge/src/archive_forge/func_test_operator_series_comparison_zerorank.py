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
def test_operator_series_comparison_zerorank(self):
    result = np.float64(0) > Series([1, 2, 3])
    expected = 0.0 > Series([1, 2, 3])
    tm.assert_series_equal(result, expected)
    result = Series([1, 2, 3]) < np.float64(0)
    expected = Series([1, 2, 3]) < 0.0
    tm.assert_series_equal(result, expected)
    result = np.array([0, 1, 2])[0] > Series([0, 1, 2])
    expected = 0.0 > Series([1, 2, 3])
    tm.assert_series_equal(result, expected)