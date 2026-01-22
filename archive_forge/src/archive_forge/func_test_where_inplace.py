import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_inplace():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0
    rs = s.copy()
    rs.where(cond, inplace=True)
    tm.assert_series_equal(rs.dropna(), s[cond])
    tm.assert_series_equal(rs, s.where(cond))
    rs = s.copy()
    rs.where(cond, -s, inplace=True)
    tm.assert_series_equal(rs, s.where(cond, -s))