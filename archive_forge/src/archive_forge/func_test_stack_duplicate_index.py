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
@pytest.mark.parametrize('idx, columns, exp_idx', [[list('abab'), ['1st', '2nd', '1st'], MultiIndex(levels=[['a', 'b'], ['1st', '2nd']], codes=[np.tile(np.arange(2).repeat(3), 2), np.tile([0, 1, 0], 4)])], [MultiIndex.from_tuples((('a', 2), ('b', 1), ('a', 1), ('b', 2))), ['1st', '2nd', '1st'], MultiIndex(levels=[['a', 'b'], [1, 2], ['1st', '2nd']], codes=[np.tile(np.arange(2).repeat(3), 2), np.repeat([1, 0, 1], [3, 6, 3]), np.tile([0, 1, 0], 4)])]])
def test_stack_duplicate_index(self, idx, columns, exp_idx, future_stack):
    df = DataFrame(np.arange(12).reshape(4, 3), index=idx, columns=columns)
    if future_stack:
        msg = 'Columns with duplicate values are not supported in stack'
        with pytest.raises(ValueError, match=msg):
            df.stack(future_stack=future_stack)
    else:
        result = df.stack(future_stack=future_stack)
        expected = Series(np.arange(12), index=exp_idx)
        tm.assert_series_equal(result, expected)
        assert result.index.is_unique is False
        li, ri = (result.index, expected.index)
        tm.assert_index_equal(li, ri)