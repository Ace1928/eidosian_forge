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
def test_df_add_2d_array_rowlike_broadcasts(self):
    arr = np.arange(6).reshape(3, 2)
    df = DataFrame(arr, columns=[True, False], index=['A', 'B', 'C'])
    rowlike = arr[[1], :]
    assert rowlike.shape == (1, df.shape[1])
    expected = DataFrame([[2, 4], [4, 6], [6, 8]], columns=df.columns, index=df.index, dtype=arr.dtype)
    result = df + rowlike
    tm.assert_frame_equal(result, expected)
    result = rowlike + df
    tm.assert_frame_equal(result, expected)