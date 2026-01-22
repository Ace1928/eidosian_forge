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
def test_iloc_setitem_list_of_lists(self):
    df = DataFrame({'A': np.arange(5, dtype='int64'), 'B': np.arange(5, 10, dtype='int64')})
    df.iloc[2:4] = [[10, 11], [12, 13]]
    expected = DataFrame({'A': [0, 1, 10, 12, 4], 'B': [5, 6, 11, 13, 9]})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'A': ['a', 'b', 'c', 'd', 'e'], 'B': np.arange(5, 10, dtype='int64')})
    df.iloc[2:4] = [['x', 11], ['y', 13]]
    expected = DataFrame({'A': ['a', 'b', 'x', 'y', 'e'], 'B': [5, 6, 11, 13, 9]})
    tm.assert_frame_equal(df, expected)