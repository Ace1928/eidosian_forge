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
@pytest.mark.parametrize('val', [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3], dtype=np.int64), range(1, 4)])
def test_alignment_non_pandas(self, val):
    index = ['A', 'B', 'C']
    columns = ['X', 'Y', 'Z']
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
    align = DataFrame._align_for_op
    expected = DataFrame({'X': val, 'Y': val, 'Z': val}, index=df.index)
    tm.assert_frame_equal(align(df, val, axis=0)[1], expected)
    expected = DataFrame({'X': [1, 1, 1], 'Y': [2, 2, 2], 'Z': [3, 3, 3]}, index=df.index)
    tm.assert_frame_equal(align(df, val, axis=1)[1], expected)