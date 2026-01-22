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
def test_string_astype(self):
    """Ticket #1748"""
    s1 = b'black'
    s2 = b'white'
    s3 = b'other'
    a = np.array([[s1], [s2], [s3]])
    assert_equal(a.dtype, np.dtype('S5'))
    b = a.astype(np.dtype('S0'))
    assert_equal(b.dtype, np.dtype('S5'))