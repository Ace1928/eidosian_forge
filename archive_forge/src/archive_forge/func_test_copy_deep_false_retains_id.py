from copy import (
import pytest
from pandas import MultiIndex
import pandas._testing as tm
def test_copy_deep_false_retains_id():
    idx = MultiIndex(levels=[['foo', 'bar'], ['fizz', 'buzz']], codes=[[0, 0, 0, 1], [0, 0, 1, 1]], names=['first', 'second'])
    res = idx.copy(deep=False)
    assert res._id is idx._id