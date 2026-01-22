import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_value_counts_integer_columns():
    df = DataFrame({1: ['a', 'a', 'a'], 2: ['a', 'a', 'd'], 3: ['a', 'b', 'c']})
    gp = df.groupby([1, 2], as_index=False, sort=False)
    result = gp[3].value_counts()
    expected = DataFrame({1: ['a', 'a', 'a'], 2: ['a', 'a', 'd'], 3: ['a', 'b', 'c'], 'count': 1})
    tm.assert_frame_equal(result, expected)