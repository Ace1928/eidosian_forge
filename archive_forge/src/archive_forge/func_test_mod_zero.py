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
def test_mod_zero(self, zero, numeric_idx):
    idx = numeric_idx
    expected = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    result = idx % zero
    tm.assert_index_equal(result, expected)
    ser_compat = Series(idx).astype('i8') % np.array(zero).astype('i8')
    tm.assert_series_equal(ser_compat, Series(result))