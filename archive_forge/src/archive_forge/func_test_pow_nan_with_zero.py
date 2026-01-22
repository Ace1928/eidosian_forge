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
def test_pow_nan_with_zero(self, box_with_array):
    left = Index([np.nan, np.nan, np.nan])
    right = Index([0, 0, 0])
    expected = Index([1.0, 1.0, 1.0])
    left = tm.box_expected(left, box_with_array)
    right = tm.box_expected(right, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = left ** right
    tm.assert_equal(result, expected)