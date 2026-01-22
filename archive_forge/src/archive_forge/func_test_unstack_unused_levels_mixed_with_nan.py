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
@pytest.mark.parametrize('level, idces, col_level, idx_level', ((0, [13, 16, 6, 9, 2, 5, 8, 11], [np.nan, 'a', 2], [np.nan, 5, 1]), (1, [8, 11, 1, 4, 12, 15, 13, 16], [np.nan, 5, 1], [np.nan, 'a', 2])))
def test_unstack_unused_levels_mixed_with_nan(self, level, idces, col_level, idx_level):
    levels = [['a', 2, 'c'], [1, 3, 5, 7]]
    codes = [[0, -1, 1, 1], [0, 2, -1, 2]]
    idx = MultiIndex(levels, codes)
    data = np.arange(8)
    df = DataFrame(data.reshape(4, 2), index=idx)
    result = df.unstack(level=level)
    exp_data = np.zeros(18) * np.nan
    exp_data[idces] = data
    cols = MultiIndex.from_product([[0, 1], col_level])
    expected = DataFrame(exp_data.reshape(3, 6), index=idx_level, columns=cols)
    tm.assert_frame_equal(result, expected)