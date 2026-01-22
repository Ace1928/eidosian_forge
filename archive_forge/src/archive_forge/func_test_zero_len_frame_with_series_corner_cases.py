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
def test_zero_len_frame_with_series_corner_cases():
    df = DataFrame(columns=['A', 'B'], dtype=np.float64)
    ser = Series([1, 2], index=['A', 'B'])
    result = df + ser
    expected = df
    tm.assert_frame_equal(result, expected)