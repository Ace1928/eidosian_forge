from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_interval(self):
    df = DataFrame({Interval(1, 2): [1, 2]})
    result = df.iloc[0]
    expected = Series({Interval(1, 2): 1}, name=0)
    tm.assert_series_equal(result, expected)
    result = df.iloc[:, 0]
    expected = Series([1, 2], name=Interval(1, 2))
    tm.assert_series_equal(result, expected)
    result = df.copy()
    result.iloc[:, 0] += 1
    expected = DataFrame({Interval(1, 2): [2, 3]})
    tm.assert_frame_equal(result, expected)