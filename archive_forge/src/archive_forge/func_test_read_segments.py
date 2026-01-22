import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_read_segments():
    fobj = BytesIO()
    arr = np.arange(100, dtype=np.int16)
    fobj.write(arr.tobytes())
    _check_bytes(read_segments(fobj, [(0, 200)], 200), arr)
    _check_bytes(read_segments(fobj, [(0, 100), (100, 100)], 200), arr)
    _check_bytes(read_segments(fobj, [(0, 50), (100, 50)], 100), np.r_[arr[:25], arr[50:75]])
    _check_bytes(read_segments(fobj, [(10, 40), (100, 50)], 90), np.r_[arr[5:25], arr[50:75]])
    _check_bytes(read_segments(fobj, [], 0), arr[0:0])
    with pytest.raises(ValueError):
        read_segments(fobj, [], 1)
    with pytest.raises(ValueError):
        read_segments(fobj, [(0, 200)], 199)
    with pytest.raises(Exception):
        read_segments(fobj, [(0, 100), (100, 200)], 199)