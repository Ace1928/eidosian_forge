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
def test_reshape_order(self):
    a = np.arange(6).reshape(2, 3, order='F')
    assert_equal(a, [[0, 2, 4], [1, 3, 5]])
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    b = a[:, 1]
    assert_equal(b.reshape(2, 2, order='F'), [[2, 6], [4, 8]])