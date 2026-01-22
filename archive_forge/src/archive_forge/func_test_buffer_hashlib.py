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
def test_buffer_hashlib(self):
    from hashlib import sha256
    x = np.array([1, 2, 3], dtype=np.dtype('<i4'))
    assert_equal(sha256(x).hexdigest(), '4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d')