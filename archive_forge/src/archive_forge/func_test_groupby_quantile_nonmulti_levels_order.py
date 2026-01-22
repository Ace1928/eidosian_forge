import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_quantile_nonmulti_levels_order():
    ind = pd.MultiIndex.from_tuples([(0, 'a', 'B'), (0, 'a', 'A'), (0, 'b', 'B'), (0, 'b', 'A'), (1, 'a', 'B'), (1, 'a', 'A'), (1, 'b', 'B'), (1, 'b', 'A')], names=['sample', 'cat0', 'cat1'])
    ser = pd.Series(range(8), index=ind)
    result = ser.groupby(level='cat1', sort=False).quantile([0.2, 0.8])
    qind = pd.MultiIndex.from_tuples([('B', 0.2), ('B', 0.8), ('A', 0.2), ('A', 0.8)], names=['cat1', None])
    expected = pd.Series([1.2, 4.8, 2.2, 5.8], index=qind)
    tm.assert_series_equal(result, expected)
    expected_levels = pd.core.indexes.frozen.FrozenList([['B', 'A'], [0.2, 0.8]])
    tm.assert_equal(result.index.levels, expected_levels)