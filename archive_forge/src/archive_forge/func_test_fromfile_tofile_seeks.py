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
def test_fromfile_tofile_seeks(self):
    f0 = tempfile.NamedTemporaryFile()
    f = f0.file
    f.write(np.arange(255, dtype='u1').tobytes())
    f.seek(20)
    ret = np.fromfile(f, count=4, dtype='u1')
    assert_equal(ret, np.array([20, 21, 22, 23], dtype='u1'))
    assert_equal(f.tell(), 24)
    f.seek(40)
    np.array([1, 2, 3], dtype='u1').tofile(f)
    assert_equal(f.tell(), 43)
    f.seek(40)
    data = f.read(3)
    assert_equal(data, b'\x01\x02\x03')
    f.seek(80)
    f.read(4)
    data = np.fromfile(f, dtype='u1', count=4)
    assert_equal(data, np.array([84, 85, 86, 87], dtype='u1'))
    f.close()