import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_fileslice_dtype():
    sliceobj = (slice(None), slice(2))
    for dt in (np.dtype('int32'), np.int32, 'i4', 'int32', '>i4', '<i4'):
        arr = np.arange(24, dtype=dt).reshape((2, 3, 4))
        fobj = BytesIO(arr.tobytes())
        new_slice = fileslice(fobj, sliceobj, arr.shape, dt)
        assert_array_equal(arr[sliceobj], new_slice)