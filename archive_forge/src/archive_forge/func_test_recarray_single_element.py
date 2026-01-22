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
def test_recarray_single_element(self):
    a = np.array([1, 2, 3], dtype=np.int32)
    b = a.copy()
    r = np.rec.array(a, shape=1, formats=['3i4'], names=['d'])
    assert_array_equal(a, b)
    assert_equal(a, r[0][0])