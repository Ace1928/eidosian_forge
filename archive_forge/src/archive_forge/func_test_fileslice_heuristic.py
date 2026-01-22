import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_fileslice_heuristic():
    shape = (15, 16, 17)
    arr = np.arange(np.prod(shape)).reshape(shape)
    for heuristic in (_always, _never, _partial, threshold_heuristic):
        for order in 'FC':
            fobj = BytesIO()
            fobj.write(arr.tobytes(order=order))
            sliceobj = (1, slice(0, 15, 2), slice(None))
            _check_slicer(sliceobj, arr, fobj, 0, order, heuristic)
            new_slice = _simple_fileslice(fobj, sliceobj, arr.shape, arr.dtype, 0, order, heuristic)
            assert_array_equal(arr[sliceobj], new_slice)