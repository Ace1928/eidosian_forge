from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_iloc_row_slice_view(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2))
    original = df.copy()
    subset = df.iloc[slice(4, 8)]
    assert np.shares_memory(df[2], subset[2])
    exp_col = original[2].copy()
    with tm.assert_cow_warning(warn_copy_on_write):
        subset.loc[:, 2] = 0.0
    if not using_copy_on_write:
        exp_col._values[4:8] = 0.0
        assert np.shares_memory(df[2], subset[2])
    tm.assert_series_equal(df[2], exp_col)