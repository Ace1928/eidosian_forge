import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('query, expected', [([Interval(2, 4, closed='right')], [1]), ([Interval(2, 4, closed='left')], [-1]), ([Interval(2, 4, closed='both')], [-1]), ([Interval(2, 4, closed='neither')], [-1]), ([Interval(1, 4, closed='right')], [-1]), ([Interval(0, 4, closed='right')], [-1]), ([Interval(0.5, 1.5, closed='right')], [-1]), ([Interval(2, 4, closed='right'), Interval(0, 1, closed='right')], [1, -1]), ([Interval(2, 4, closed='right'), Interval(2, 4, closed='right')], [1, 1]), ([Interval(5, 7, closed='right'), Interval(2, 4, closed='right')], [2, 1]), ([Interval(2, 4, closed='right'), Interval(2, 4, closed='left')], [1, -1])])
def test_get_indexer_with_interval(self, query, expected):
    tuples = [(0, 2), (2, 4), (5, 7)]
    index = IntervalIndex.from_tuples(tuples, closed='right')
    result = index.get_indexer(query)
    expected = np.array(expected, dtype='intp')
    tm.assert_numpy_array_equal(result, expected)