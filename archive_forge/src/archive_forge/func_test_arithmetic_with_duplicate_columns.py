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
@pytest.mark.parametrize('op', ['__add__', '__mul__', '__sub__', '__truediv__'])
def test_arithmetic_with_duplicate_columns(self, op):
    df = DataFrame({'A': np.arange(10), 'B': np.random.default_rng(2).random(10)})
    expected = getattr(df, op)(df)
    expected.columns = ['A', 'A']
    df.columns = ['A', 'A']
    result = getattr(df, op)(df)
    tm.assert_frame_equal(result, expected)