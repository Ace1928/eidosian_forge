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
@pytest.mark.parametrize('val', [np.ones((10, 10), dtype='int32'), np.uint64(10)])
@pytest.mark.parametrize('protocol', range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_module(self, protocol, val):
    s = pickle.dumps(val, protocol)
    assert b'_multiarray_umath' not in s
    if protocol == 5 and len(val.shape) > 0:
        assert b'numpy.core.numeric' in s
    else:
        assert b'numpy.core.multiarray' in s