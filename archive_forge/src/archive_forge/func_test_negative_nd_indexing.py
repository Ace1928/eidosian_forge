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
def test_negative_nd_indexing(self):
    c = np.arange(125).reshape((5, 5, 5))
    origidx = np.array([-1, 0, 1])
    idx = np.array(origidx)
    c[idx]
    assert_array_equal(idx, origidx)