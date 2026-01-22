import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_26395(indexer_al):
    df = DataFrame(index=['A', 'B', 'C'])
    df['D'] = 0
    indexer_al(df)['C', 'D'] = 2
    expected = DataFrame({'D': [0, 0, 2]}, index=['A', 'B', 'C'], dtype=np.int64)
    tm.assert_frame_equal(df, expected)
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        indexer_al(df)['C', 'D'] = 44.5
    expected = DataFrame({'D': [0, 0, 44.5]}, index=['A', 'B', 'C'], dtype=np.float64)
    tm.assert_frame_equal(df, expected)
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        indexer_al(df)['C', 'D'] = 'hello'
    expected = DataFrame({'D': [0, 0, 'hello']}, index=['A', 'B', 'C'], dtype=object)
    tm.assert_frame_equal(df, expected)