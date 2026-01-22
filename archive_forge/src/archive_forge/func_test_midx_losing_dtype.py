from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_midx_losing_dtype():
    midx = MultiIndex.from_arrays([[0, 0], [np.nan, np.nan]])
    midx2 = MultiIndex.from_arrays([[1, 1], [np.nan, np.nan]])
    df1 = DataFrame({'a': [None, 4]}, index=midx)
    df2 = DataFrame({'a': [3, 3]}, index=midx2)
    result = df1.combine_first(df2)
    expected_midx = MultiIndex.from_arrays([[0, 0, 1, 1], [np.nan, np.nan, np.nan, np.nan]])
    expected = DataFrame({'a': [np.nan, 4, 3, 3]}, index=expected_midx)
    tm.assert_frame_equal(result, expected)