import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_deepcopy_on_0d_array(self):
    arr = np.array(3)
    arr_cp = copy.deepcopy(arr)
    assert_equal(arr, arr_cp)
    assert_equal(arr.shape, arr_cp.shape)
    assert_equal(int(arr), int(arr_cp))
    assert_(arr is not arr_cp)
    assert_(isinstance(arr_cp, type(arr)))