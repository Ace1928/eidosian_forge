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
def test_series_divmod_zero(self):
    tser = Series(np.arange(1, 11, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    other = tser * 0
    result = divmod(tser, other)
    exp1 = Series([np.inf] * len(tser), index=tser.index, name='ts')
    exp2 = Series([np.nan] * len(tser), index=tser.index, name='ts')
    tm.assert_series_equal(result[0], exp1)
    tm.assert_series_equal(result[1], exp2)