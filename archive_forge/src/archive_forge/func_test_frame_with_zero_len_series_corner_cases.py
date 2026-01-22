from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_frame_with_zero_len_series_corner_cases():
    df = DataFrame(np.random.default_rng(2).standard_normal(6).reshape(3, 2), columns=['A', 'B'])
    ser = Series(dtype=np.float64)
    result = df + ser
    expected = DataFrame(df.values * np.nan, columns=df.columns)
    tm.assert_frame_equal(result, expected)
    with pytest.raises(ValueError, match='not aligned'):
        df == ser
    df2 = DataFrame(df.values.view('M8[ns]'), columns=df.columns)
    with pytest.raises(ValueError, match='not aligned'):
        df2 == ser