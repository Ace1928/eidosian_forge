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
def test_void_scalar_constructor(self):
    test_string = np.array('test')
    test_string_void_scalar = np.core.multiarray.scalar(np.dtype(('V', test_string.dtype.itemsize)), test_string.tobytes())
    assert_(test_string_void_scalar.view(test_string.dtype) == test_string)
    test_record = np.ones((), 'i,i')
    test_record_void_scalar = np.core.multiarray.scalar(test_record.dtype, test_record.tobytes())
    assert_(test_record_void_scalar == test_record)
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        assert_(pickle.loads(pickle.dumps(test_string, protocol=proto)) == test_string)
        assert_(pickle.loads(pickle.dumps(test_record, protocol=proto)) == test_record)