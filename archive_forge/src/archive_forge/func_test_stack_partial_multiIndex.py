from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
@pytest.mark.parametrize('multiindex_columns', [[0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 1], [0, 2], [0, 3], [0], [2], [4], [4, 3, 2, 1, 0], [3, 2, 1, 0], [4, 2, 1, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2], [1, 0], [2, 0], [3, 0]])
@pytest.mark.parametrize('level', (-1, 0, 1, [0, 1], [1, 0]))
def test_stack_partial_multiIndex(self, multiindex_columns, level, future_stack):
    dropna = False if not future_stack else lib.no_default
    full_multiindex = MultiIndex.from_tuples([('B', 'x'), ('B', 'z'), ('A', 'y'), ('C', 'x'), ('C', 'u')], names=['Upper', 'Lower'])
    multiindex = full_multiindex[multiindex_columns]
    df = DataFrame(np.arange(3 * len(multiindex)).reshape(3, len(multiindex)), columns=multiindex)
    result = df.stack(level=level, dropna=dropna, future_stack=future_stack)
    if isinstance(level, int) and (not future_stack):
        expected = df.stack(level=level, dropna=True, future_stack=future_stack)
        if isinstance(expected, Series):
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)
    df.columns = MultiIndex.from_tuples(df.columns.to_numpy(), names=df.columns.names)
    expected = df.stack(level=level, dropna=dropna, future_stack=future_stack)
    if isinstance(expected, Series):
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_frame_equal(result, expected)