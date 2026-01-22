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
def test_fields_strides(self):
    """gh-2355"""
    r = np.frombuffer(b'abcdefghijklmnop' * 4 * 3, dtype='i4,(2,3)u2')
    assert_equal(r[0:3:2]['f1'], r['f1'][0:3:2])
    assert_equal(r[0:3:2]['f1'][0], r[0:3:2][0]['f1'])
    assert_equal(r[0:3:2]['f1'][0][()], r[0:3:2][0]['f1'][()])
    assert_equal(r[0:3:2]['f1'][0].strides, r[0:3:2][0]['f1'].strides)