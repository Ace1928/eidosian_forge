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
@pytest.mark.parametrize('other', [np.nan, 7, -23, 2.718, -3.14, np.inf])
def test_ops_np_scalar(self, other):
    vals = np.random.default_rng(2).standard_normal((5, 3))
    f = lambda x: pd.DataFrame(x, index=list('ABCDE'), columns=['jim', 'joe', 'jolie'])
    df = f(vals)
    tm.assert_frame_equal(df / np.array(other), f(vals / other))
    tm.assert_frame_equal(np.array(other) * df, f(vals * other))
    tm.assert_frame_equal(df + np.array(other), f(vals + other))
    tm.assert_frame_equal(np.array(other) - df, f(other - vals))