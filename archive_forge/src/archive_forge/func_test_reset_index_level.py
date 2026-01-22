from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('levels', [['A', 'B'], [0, 1]])
def test_reset_index_level(self, levels):
    df = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=['A', 'B', 'C', 'D'])
    result = df.set_index(['A', 'B']).reset_index(level=levels[0])
    tm.assert_frame_equal(result, df.set_index('B'))
    result = df.set_index(['A', 'B']).reset_index(level=levels[:1])
    tm.assert_frame_equal(result, df.set_index('B'))
    result = df.set_index(['A', 'B']).reset_index(level=levels)
    tm.assert_frame_equal(result, df)
    result = df.set_index(['A', 'B']).reset_index(level=levels, drop=True)
    tm.assert_frame_equal(result, df[['C', 'D']])
    result = df.set_index('A').reset_index(level=levels[0])
    tm.assert_frame_equal(result, df)
    result = df.set_index('A').reset_index(level=levels[:1])
    tm.assert_frame_equal(result, df)
    result = df.set_index(['A']).reset_index(level=levels[0], drop=True)
    tm.assert_frame_equal(result, df[['B', 'C', 'D']])