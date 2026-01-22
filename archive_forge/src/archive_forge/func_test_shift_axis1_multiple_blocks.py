import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_axis1_multiple_blocks(self, using_array_manager):
    df1 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 3)))
    df2 = DataFrame(np.random.default_rng(2).integers(1000, size=(5, 2)))
    df3 = pd.concat([df1, df2], axis=1)
    if not using_array_manager:
        assert len(df3._mgr.blocks) == 2
    result = df3.shift(2, axis=1)
    expected = df3.take([-1, -1, 0, 1, 2], axis=1)
    expected = expected.pipe(lambda df: df.set_axis(range(df.shape[1]), axis=1).astype({0: 'float', 1: 'float'}).set_axis(df.columns, axis=1))
    expected.iloc[:, :2] = np.nan
    expected.columns = df3.columns
    tm.assert_frame_equal(result, expected)
    df3 = pd.concat([df1, df2], axis=1)
    if not using_array_manager:
        assert len(df3._mgr.blocks) == 2
    result = df3.shift(-2, axis=1)
    expected = df3.take([2, 3, 4, -1, -1], axis=1)
    expected = expected.pipe(lambda df: df.set_axis(range(df.shape[1]), axis=1).astype({3: 'float', 4: 'float'}).set_axis(df.columns, axis=1))
    expected.iloc[:, -2:] = np.nan
    expected.columns = df3.columns
    tm.assert_frame_equal(result, expected)