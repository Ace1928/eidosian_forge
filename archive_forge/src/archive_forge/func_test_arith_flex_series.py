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
def test_arith_flex_series(self, simple_frame):
    df = simple_frame
    row = df.xs('a')
    col = df['two']
    tm.assert_frame_equal(df.add(row, axis=None), df + row)
    tm.assert_frame_equal(df.div(row), df / row)
    tm.assert_frame_equal(df.div(col, axis=0), (df.T / col).T)