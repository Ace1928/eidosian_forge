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
def test_frompyfunc_nout_0(self):

    def f(x):
        x[0], x[-1] = (x[-1], x[0])
    uf = np.frompyfunc(f, 1, 0)
    a = np.array([[1, 2, 3], [4, 5], [6, 7, 8, 9]], dtype=object)
    assert_equal(uf(a), ())
    expected = np.array([[3, 2, 1], [5, 4], [9, 7, 8, 6]], dtype=object)
    assert_array_equal(a, expected)