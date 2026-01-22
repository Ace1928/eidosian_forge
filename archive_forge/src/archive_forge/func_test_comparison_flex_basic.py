from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('axis', [0, None, 'index'])
def test_comparison_flex_basic(self, axis, comparison_op):
    left = Series(np.random.default_rng(2).standard_normal(10))
    right = Series(np.random.default_rng(2).standard_normal(10))
    result = getattr(left, comparison_op.__name__)(right, axis=axis)
    expected = comparison_op(left, right)
    tm.assert_series_equal(result, expected)