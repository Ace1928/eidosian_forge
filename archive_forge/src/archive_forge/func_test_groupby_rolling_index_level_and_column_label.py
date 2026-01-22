import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_index_level_and_column_label(self):
    arrays = [['val1', 'val1', 'val2'], ['val1', 'val1', 'val2']]
    index = MultiIndex.from_arrays(arrays, names=('idx1', 'idx2'))
    df = DataFrame({'A': [1, 1, 2], 'B': range(3)}, index=index)
    result = df.groupby(['idx1', 'A']).rolling(1).mean()
    expected = DataFrame({'B': [0.0, 1.0, 2.0]}, index=MultiIndex.from_tuples([('val1', 1, 'val1', 'val1'), ('val1', 1, 'val1', 'val1'), ('val2', 2, 'val2', 'val2')], names=['idx1', 'A', 'idx1', 'idx2']))
    tm.assert_frame_equal(result, expected)