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
@pytest.mark.parametrize('dtype', ['int64', 'float64'])
def test_arith_flex_series_broadcasting(self, dtype):
    df = DataFrame(np.arange(3 * 2).reshape((3, 2)), dtype=dtype)
    expected = DataFrame([[np.nan, np.inf], [1.0, 1.5], [1.0, 1.25]])
    result = df.div(df[0], axis='index')
    tm.assert_frame_equal(result, expected)