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
def test_inplace_arithmetic_series_update(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': [1, 2, 3]})
    df_orig = df.copy()
    series = df['A']
    vals = series._values
    with tm.assert_cow_warning(warn_copy_on_write):
        series += 1
    if using_copy_on_write:
        assert series._values is not vals
        tm.assert_frame_equal(df, df_orig)
    else:
        assert series._values is vals
        expected = DataFrame({'A': [2, 3, 4]})
        tm.assert_frame_equal(df, expected)