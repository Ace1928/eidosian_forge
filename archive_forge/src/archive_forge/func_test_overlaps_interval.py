import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_overlaps_interval(self, constructor, start_shift, closed, other_closed):
    start, shift = start_shift
    interval = Interval(start, start + 3 * shift, other_closed)
    tuples = [(start, start + 3 * shift), (start + shift, start + 2 * shift), (start - shift, start + 4 * shift), (start + 2 * shift, start + 4 * shift), (start + 3 * shift, start + 4 * shift), (start + 4 * shift, start + 5 * shift)]
    interval_container = constructor.from_tuples(tuples, closed)
    adjacent = interval.closed_right and interval_container.closed_left
    expected = np.array([True, True, True, True, adjacent, False])
    result = interval_container.overlaps(interval)
    tm.assert_numpy_array_equal(result, expected)