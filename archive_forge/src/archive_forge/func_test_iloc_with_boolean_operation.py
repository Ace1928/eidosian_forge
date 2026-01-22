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
def test_iloc_with_boolean_operation(self):
    result = DataFrame([[0, 1], [2, 3], [4, 5], [6, np.nan]])
    result.iloc[result.index <= 2] *= 2
    expected = DataFrame([[0, 2], [4, 6], [8, 10], [6, np.nan]])
    tm.assert_frame_equal(result, expected)
    result.iloc[result.index > 2] *= 2
    expected = DataFrame([[0, 2], [4, 6], [8, 10], [12, np.nan]])
    tm.assert_frame_equal(result, expected)
    result.iloc[[True, True, False, False]] *= 2
    expected = DataFrame([[0, 4], [8, 12], [8, 10], [12, np.nan]])
    tm.assert_frame_equal(result, expected)
    result.iloc[[False, False, True, True]] /= 2
    expected = DataFrame([[0, 4.0], [8, 12.0], [4, 5.0], [6, np.nan]])
    tm.assert_frame_equal(result, expected)