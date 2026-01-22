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
def test_df_mod_zero_df(self, using_array_manager):
    df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]})
    first = Series([0, 0, 0, 0])
    if not using_array_manager:
        first = first.astype('float64')
    second = Series([np.nan, np.nan, np.nan, 0])
    expected = pd.DataFrame({'first': first, 'second': second})
    result = df % df
    tm.assert_frame_equal(result, expected)
    df = pd.DataFrame({'first': [3, 4, 5, 8], 'second': [0, 0, 0, 3]}, copy=False)
    first = Series([0, 0, 0, 0], dtype='int64')
    second = Series([np.nan, np.nan, np.nan, 0])
    expected = pd.DataFrame({'first': first, 'second': second})
    result = df % df
    tm.assert_frame_equal(result, expected)