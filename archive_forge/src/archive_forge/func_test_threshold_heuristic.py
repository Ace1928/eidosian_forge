import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_threshold_heuristic():
    assert threshold_heuristic(1, 9, 1, skip_thresh=8) == 'full'
    assert threshold_heuristic(1, 9, 1, skip_thresh=7) is None
    assert threshold_heuristic(1, 9, 2, skip_thresh=16) == 'full'
    assert threshold_heuristic(1, 9, 2, skip_thresh=15) is None
    assert threshold_heuristic(slice(0, 9, 1), 9, 2, skip_thresh=2) == 'full'
    assert threshold_heuristic(slice(0, 9, 1), 9, 2, skip_thresh=1) == None
    assert threshold_heuristic(slice(0, 9, 2), 9, 2, skip_thresh=3) == None
    assert threshold_heuristic(slice(9, None, -1), 9, 2, skip_thresh=2) == 'full'
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=2) == 'contiguous'
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=1) == None
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=4) == 'full'
    assert threshold_heuristic(slice(2, 9, 1), 9, 2, skip_thresh=3) == 'contiguous'