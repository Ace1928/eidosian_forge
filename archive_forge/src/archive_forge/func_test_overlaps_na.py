import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_overlaps_na(self, constructor, start_shift):
    """NA values are marked as False"""
    start, shift = start_shift
    interval = Interval(start, start + shift)
    tuples = [(start, start + shift), np.nan, (start + 2 * shift, start + 3 * shift)]
    interval_container = constructor.from_tuples(tuples)
    expected = np.array([True, False, False])
    result = interval_container.overlaps(interval)
    tm.assert_numpy_array_equal(result, expected)