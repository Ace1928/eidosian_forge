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
@pytest.mark.parametrize('values', [[1, 2], (1, 2), np.array([1, 2]), range(1, 3), deque([1, 2])])
def test_arith_alignment_non_pandas_object(self, values):
    df = DataFrame({'A': [1, 1], 'B': [1, 1]})
    expected = DataFrame({'A': [2, 2], 'B': [3, 3]})
    result = df + values
    tm.assert_frame_equal(result, expected)