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
def test_pickle_datetime64_array(self):
    d = np.datetime64('2015-07-04 12:59:59.50', 'ns')
    arr = np.array([d])
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        dumped = pickle.dumps(arr, protocol=proto)
        assert_equal(pickle.loads(dumped), arr)