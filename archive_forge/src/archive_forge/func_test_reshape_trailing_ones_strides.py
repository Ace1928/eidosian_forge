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
@pytest.mark.skipif(np.ones(1).strides[0] == np.iinfo(np.intp).max, reason='Using relaxed stride debug')
def test_reshape_trailing_ones_strides(self):
    a = np.zeros(12, dtype=np.int32)[::2]
    strides_c = (16, 8, 8, 8)
    strides_f = (8, 24, 48, 48)
    assert_equal(a.reshape(3, 2, 1, 1).strides, strides_c)
    assert_equal(a.reshape(3, 2, 1, 1, order='F').strides, strides_f)
    assert_equal(np.array(0, dtype=np.int32).reshape(1, 1).strides, (4, 4))