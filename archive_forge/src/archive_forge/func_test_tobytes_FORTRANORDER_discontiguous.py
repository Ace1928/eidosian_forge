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
def test_tobytes_FORTRANORDER_discontiguous(self):
    x = np.array(np.random.rand(3, 3), order='F')[:, :2]
    assert_array_almost_equal(x.ravel(), np.frombuffer(x.tobytes()))