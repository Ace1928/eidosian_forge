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
def test_rdiv_zero_compat(self):
    zero_array = np.array([0] * 5)
    data = np.random.default_rng(2).standard_normal(5)
    expected = Series([0.0] * 5)
    result = zero_array / Series(data)
    tm.assert_series_equal(result, expected)
    result = Series(zero_array) / data
    tm.assert_series_equal(result, expected)
    result = Series(zero_array) / Series(data)
    tm.assert_series_equal(result, expected)