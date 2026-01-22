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
def test_df_mod_zero_series_does_not_commute(self):
    df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    ser = df[0]
    res = ser % df
    res2 = df % ser
    assert not res.fillna(0).equals(res2.fillna(0))