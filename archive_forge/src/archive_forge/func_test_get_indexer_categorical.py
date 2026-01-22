import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('target', [IntervalIndex.from_tuples([(7, 8), (1, 2), (3, 4), (0, 1)]), IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4), np.nan]), IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)], closed='both'), [-1, 0, 0.5, 1, 2, 2.5, np.nan], ['foo', 'foo', 'bar', 'baz']])
def test_get_indexer_categorical(self, target, ordered):
    index = IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)])
    categorical_target = CategoricalIndex(target, ordered=ordered)
    result = index.get_indexer(categorical_target)
    expected = index.get_indexer(target)
    tm.assert_numpy_array_equal(result, expected)