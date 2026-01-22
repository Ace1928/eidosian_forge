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
@pytest.mark.parametrize('power', [1, 2, 5])
@pytest.mark.parametrize('string_size', [0, 1, 2, 5])
def test_empty_str_comparison(power, string_size):
    a = np.array(range(10 ** power))
    right = pd.DataFrame(a, dtype=np.int64)
    left = ' ' * string_size
    result = right == left
    expected = pd.DataFrame(np.zeros(right.shape, dtype=bool))
    tm.assert_frame_equal(result, expected)