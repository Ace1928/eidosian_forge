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
def test_floordiv_axis0(self):
    arr = np.arange(3)
    ser = Series(arr)
    df = DataFrame({'A': ser, 'B': ser})
    result = df.floordiv(ser, axis=0)
    expected = DataFrame({col: df[col] // ser for col in df.columns})
    tm.assert_frame_equal(result, expected)
    result2 = df.floordiv(ser.values, axis=0)
    tm.assert_frame_equal(result2, expected)