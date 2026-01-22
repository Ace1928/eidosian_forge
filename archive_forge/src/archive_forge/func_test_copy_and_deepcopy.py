from copy import (
import pytest
from pandas import MultiIndex
import pandas._testing as tm
@pytest.mark.parametrize('func', [copy, deepcopy])
def test_copy_and_deepcopy(func):
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    idx_copy = func(idx)
    assert idx_copy is not idx
    assert idx_copy.equals(idx)