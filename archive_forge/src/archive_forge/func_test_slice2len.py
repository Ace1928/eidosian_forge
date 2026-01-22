import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_slice2len():
    assert slice2len(slice(None), 10) == 10
    assert slice2len(slice(11), 10) == 10
    assert slice2len(slice(1, 11), 10) == 9
    assert slice2len(slice(1, 1), 10) == 0
    assert slice2len(slice(1, 11, 2), 10) == 5
    assert slice2len(slice(0, 11, 3), 10) == 4
    assert slice2len(slice(1, 11, 3), 10) == 3
    assert slice2len(slice(None, None, -1), 10) == 10
    assert slice2len(slice(11, None, -1), 10) == 10
    assert slice2len(slice(None, 1, -1), 10) == 8
    assert slice2len(slice(None, None, -2), 10) == 5
    assert slice2len(slice(None, None, -3), 10) == 4
    assert slice2len(slice(None, 0, -3), 10) == 3
    assert slice2len(slice(None, -4, -1), 10) == 3
    assert slice2len(slice(-4, -2, 1), 10) == 2
    assert slice2len(slice(3, 2, 1), 10) == 0
    assert slice2len(slice(2, 3, -1), 10) == 0