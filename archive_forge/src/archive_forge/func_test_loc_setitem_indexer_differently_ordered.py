import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_indexer_differently_ordered():
    mi = MultiIndex.from_product([['a', 'b'], [0, 1]])
    df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=mi)
    indexer = ('a', [1, 0])
    df.loc[indexer, :] = np.array([[9, 10], [11, 12]])
    expected = DataFrame([[11, 12], [9, 10], [5, 6], [7, 8]], index=mi)
    tm.assert_frame_equal(df, expected)