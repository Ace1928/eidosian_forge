import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
def test_sortlevel_deterministic():
    tuples = [('bar', 'one'), ('foo', 'two'), ('qux', 'two'), ('foo', 'one'), ('baz', 'two'), ('qux', 'one')]
    index = MultiIndex.from_tuples(tuples)
    sorted_idx, _ = index.sortlevel(0)
    expected = MultiIndex.from_tuples(sorted(tuples))
    assert sorted_idx.equals(expected)
    sorted_idx, _ = index.sortlevel(0, ascending=False)
    assert sorted_idx.equals(expected[::-1])
    sorted_idx, _ = index.sortlevel(1)
    by1 = sorted(tuples, key=lambda x: (x[1], x[0]))
    expected = MultiIndex.from_tuples(by1)
    assert sorted_idx.equals(expected)
    sorted_idx, _ = index.sortlevel(1, ascending=False)
    assert sorted_idx.equals(expected[::-1])