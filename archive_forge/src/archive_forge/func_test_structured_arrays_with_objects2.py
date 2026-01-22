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
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='Python 3.12 has immortal refcounts, this test no longer works.')
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_structured_arrays_with_objects2(self):
    stra = 'aaaa'
    strb = 'bbbb'
    numb = sys.getrefcount(strb)
    numa = sys.getrefcount(stra)
    x = np.array([[(0, stra), (1, strb)]], 'i8,O')
    x[x.nonzero()] = x.ravel()[:1]
    assert_(sys.getrefcount(strb) == numb)
    assert_(sys.getrefcount(stra) == numa + 2)