from copy import (
import pytest
from pandas import MultiIndex
import pandas._testing as tm
@pytest.mark.parametrize('deep', [True, False])
def test_copy_method(deep):
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    idx_copy = idx.copy(deep=deep)
    assert idx_copy.equals(idx)