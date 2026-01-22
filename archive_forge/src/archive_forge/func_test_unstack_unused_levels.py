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
def test_unstack_unused_levels(self):
    idx = MultiIndex.from_product([['a'], ['A', 'B', 'C', 'D']])[:-1]
    df = DataFrame([[1, 0]] * 3, index=idx)
    result = df.unstack()
    exp_col = MultiIndex.from_product([[0, 1], ['A', 'B', 'C']])
    expected = DataFrame([[1, 1, 1, 0, 0, 0]], index=['a'], columns=exp_col)
    tm.assert_frame_equal(result, expected)
    assert (result.columns.levels[1] == idx.levels[1]).all()
    levels = [[0, 1, 7], [0, 1, 2, 3]]
    codes = [[0, 0, 1, 1], [0, 2, 0, 2]]
    idx = MultiIndex(levels, codes)
    block = np.arange(4).reshape(2, 2)
    df = DataFrame(np.concatenate([block, block + 4]), index=idx)
    result = df.unstack()
    expected = DataFrame(np.concatenate([block * 2, block * 2 + 1], axis=1), columns=idx)
    tm.assert_frame_equal(result, expected)
    assert (result.columns.levels[1] == idx.levels[1]).all()