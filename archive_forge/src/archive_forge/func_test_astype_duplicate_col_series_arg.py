import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_duplicate_col_series_arg(self):
    vals = np.random.default_rng(2).standard_normal((3, 4))
    df = DataFrame(vals, columns=['A', 'B', 'C', 'A'])
    dtypes = df.dtypes
    dtypes.iloc[0] = str
    dtypes.iloc[2] = 'Float64'
    result = df.astype(dtypes)
    expected = DataFrame({0: Series(vals[:, 0].astype(str), dtype=object), 1: vals[:, 1], 2: pd.array(vals[:, 2], dtype='Float64'), 3: vals[:, 3]})
    expected.columns = df.columns
    tm.assert_frame_equal(result, expected)