import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
@pytest.mark.parametrize('level0', [['a', 'd', 'b'], ['a', 'd', 'b', 'unused']])
@pytest.mark.parametrize('level1', [['w', 'x', 'y', 'z'], ['w', 'x', 'y', 'z', 'unused']])
def test_remove_unused_nan(level0, level1):
    mi = MultiIndex(levels=[level0, level1], codes=[[0, 2, -1, 1, -1], [0, 1, 2, 3, 2]])
    result = mi.remove_unused_levels()
    tm.assert_index_equal(result, mi)
    for level in (0, 1):
        assert 'unused' not in result.levels[level]