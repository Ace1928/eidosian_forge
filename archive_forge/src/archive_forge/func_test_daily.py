from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_daily(self):
    rng = date_range('1/1/2000', '12/31/2004', freq='D')
    ts = Series(np.arange(len(rng)), index=rng)
    result = pivot_table(DataFrame(ts), index=ts.index.year, columns=ts.index.dayofyear)
    result.columns = result.columns.droplevel(0)
    doy = np.asarray(ts.index.dayofyear)
    expected = {}
    for y in ts.index.year.unique().values:
        mask = ts.index.year == y
        expected[y] = Series(ts.values[mask], index=doy[mask])
    expected = DataFrame(expected, dtype=float).T
    tm.assert_frame_equal(result, expected)