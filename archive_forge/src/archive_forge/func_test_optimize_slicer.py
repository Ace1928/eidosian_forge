import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_optimize_slicer():
    for all_full in (True, False):
        for heuristic in (_always, _never, _partial):
            for is_slowest in (True, False):
                assert optimize_slicer(slice(None), 10, all_full, is_slowest, 4, heuristic) == (slice(None), slice(None))
                assert optimize_slicer(slice(10), 10, all_full, is_slowest, 4, heuristic) == (slice(None), slice(None))
                assert optimize_slicer(slice(0, 10), 10, all_full, is_slowest, 4, heuristic) == (slice(None), slice(None))
                assert optimize_slicer(slice(0, 10, 1), 10, all_full, is_slowest, 4, heuristic) == (slice(None), slice(None))
                assert optimize_slicer(slice(None, None, -1), 10, all_full, is_slowest, 4, heuristic) == (slice(None), slice(None, None, -1))
    assert optimize_slicer(slice(9), 10, False, False, 4, _always) == (slice(0, 9, 1), slice(None))
    assert optimize_slicer(slice(9), 10, True, False, 4, _always) == (slice(None), slice(0, 9, 1))
    assert optimize_slicer(slice(9), 10, True, True, 4, _always) == (slice(0, 9, 1), slice(None))
    assert optimize_slicer(slice(9), 10, True, False, 4, _never) == (slice(0, 9, 1), slice(None))
    assert optimize_slicer(slice(1, 10), 10, True, False, 4, _never) == (slice(1, 10, 1), slice(None))
    assert optimize_slicer(slice(8, None, -1), 10, False, False, 4, _never) == (slice(0, 9, 1), slice(None, None, -1))
    assert optimize_slicer(slice(8, None, -1), 10, True, False, 4, _always) == (slice(None), slice(8, None, -1))
    assert optimize_slicer(slice(8, None, -1), 10, False, False, 4, _never) == (slice(0, 9, 1), slice(None, None, -1))
    assert optimize_slicer(slice(9, 0, -1), 10, False, False, 4, _never) == (slice(1, 10, 1), slice(None, None, -1))
    assert optimize_slicer(slice(0, 10, 2), 10, False, False, 4, _never) == (slice(0, 10, 2), slice(None))
    assert optimize_slicer(slice(0, 10, 2), 10, True, False, 4, _never) == (slice(0, 10, 2), slice(None))
    assert optimize_slicer(slice(0, 10, 2), 10, True, False, 4, _always) == (slice(None), slice(0, 10, 2))
    assert optimize_slicer(slice(0, 10, 2), 10, False, False, 4, _always) == (slice(0, 10, 2), slice(None))
    assert optimize_slicer(slice(10, None, -2), 10, False, False, 4, _never) == (slice(1, 10, 2), slice(None, None, -1))
    assert optimize_slicer(slice(10, None, -2), 10, True, False, 4, _always) == (slice(None), slice(9, None, -2))
    assert optimize_slicer(slice(2, 8, 2), 10, False, False, 4, _never) == (slice(2, 8, 2), slice(None))
    assert optimize_slicer(slice(2, 8, 2), 10, True, False, 4, _partial) == (slice(2, 8, 1), slice(None, None, 2))
    assert optimize_slicer(slice(0, 10, 2), 10, True, False, 4, _always) == (slice(None), slice(0, 10, 2))
    assert optimize_slicer(slice(0, 10, 2), 10, True, True, 4, _always) == (slice(0, 10, 1), slice(None, None, 2))
    assert optimize_slicer(slice(9), 10, True, True, 4, _always) == (slice(0, 9, 1), slice(None))
    assert optimize_slicer(0, 10, True, False, 4, _never) == (0, 'dropped')
    assert optimize_slicer(-1, 10, True, False, 4, _never) == (9, 'dropped')
    assert optimize_slicer(0.9, 10, True, False, 4, _never) == (0, 'dropped')
    with pytest.raises(ValueError):
        optimize_slicer(0, 10, True, False, 4, _partial)
    assert optimize_slicer(0, 10, True, False, 4, _always) == (slice(None), 0)
    assert optimize_slicer(0, 10, True, True, 4, _always) == (0, 'dropped')