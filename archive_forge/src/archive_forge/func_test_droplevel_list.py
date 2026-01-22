import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_droplevel_list():
    index = MultiIndex(levels=[Index(range(4)), Index(range(4)), Index(range(4))], codes=[np.array([0, 0, 1, 2, 2, 2, 3, 3]), np.array([0, 1, 0, 0, 0, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0, 1, 0])], names=['one', 'two', 'three'])
    dropped = index[:2].droplevel(['three', 'one'])
    expected = index[:2].droplevel(2).droplevel(0)
    assert dropped.equals(expected)
    dropped = index[:2].droplevel([])
    expected = index[:2]
    assert dropped.equals(expected)
    msg = 'Cannot remove 3 levels from an index with 3 levels: at least one level must be left'
    with pytest.raises(ValueError, match=msg):
        index[:2].droplevel(['one', 'two', 'three'])
    with pytest.raises(KeyError, match="'Level four not found'"):
        index[:2].droplevel(['one', 'four'])