import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_slicers2segments():
    assert slicers2segments((0,), (10,), 7, 4) == [[7, 4]]
    assert slicers2segments((0, 1), (10, 6), 7, 4) == [[7 + 10 * 4, 4]]
    assert slicers2segments((0, 1, 2), (10, 6, 4), 7, 4) == [[7 + 10 * 4 + 10 * 6 * 2 * 4, 4]]
    assert slicers2segments((slice(None),), (10,), 7, 4) == [[7, 10 * 4]]
    assert slicers2segments((0, slice(None)), (10, 6), 7, 4) == [[7 + 10 * 4 * i, 4] for i in range(6)]
    assert slicers2segments((slice(None), 0), (10, 6), 7, 4) == [[7, 10 * 4]]
    assert slicers2segments((slice(None), slice(None)), (10, 6), 7, 4) == [[7, 10 * 6 * 4]]
    assert slicers2segments((slice(None), slice(None), 2), (10, 6, 4), 7, 4) == [[7 + 10 * 6 * 2 * 4, 10 * 6 * 4]]