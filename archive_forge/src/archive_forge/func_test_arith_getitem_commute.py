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
@pytest.mark.parametrize('col', ['A', 'B'])
def test_arith_getitem_commute(self, all_arithmetic_functions, col):
    df = DataFrame({'A': [1.1, 3.3], 'B': [2.5, -3.9]})
    result = all_arithmetic_functions(df, 1)[col]
    expected = all_arithmetic_functions(df[col], 1)
    tm.assert_series_equal(result, expected)