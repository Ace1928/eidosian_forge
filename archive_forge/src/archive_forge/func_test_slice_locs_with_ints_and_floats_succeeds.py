import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_slice_locs_with_ints_and_floats_succeeds(self):
    index = IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)])
    assert index.slice_locs(0, 1) == (0, 1)
    assert index.slice_locs(0, 2) == (0, 2)
    assert index.slice_locs(0, 3) == (0, 2)
    assert index.slice_locs(3, 1) == (2, 1)
    assert index.slice_locs(3, 4) == (2, 3)
    assert index.slice_locs(0, 4) == (0, 3)
    index = IntervalIndex.from_tuples([(3, 4), (1, 2), (0, 1)])
    assert index.slice_locs(0, 1) == (3, 3)
    assert index.slice_locs(0, 2) == (3, 2)
    assert index.slice_locs(0, 3) == (3, 1)
    assert index.slice_locs(3, 1) == (1, 3)
    assert index.slice_locs(3, 4) == (1, 1)
    assert index.slice_locs(0, 4) == (3, 1)