from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_preserve_dtypes():
    a_column = Series(['a', 'b'], index=range(2))
    b_column = Series(range(2), index=range(2))
    df1 = DataFrame({'A': a_column, 'B': b_column})
    c_column = Series(['a', 'b'], index=range(5, 7))
    b_column = Series(range(-1, 1), index=range(5, 7))
    df2 = DataFrame({'B': b_column, 'C': c_column})
    expected = DataFrame({'A': ['a', 'b', np.nan, np.nan], 'B': [0, 1, -1, 0], 'C': [np.nan, np.nan, 'a', 'b']}, index=[0, 1, 5, 6])
    combined = df1.combine_first(df2)
    tm.assert_frame_equal(combined, expected)