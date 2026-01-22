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
def test_array_side_effect(self):
    assert_equal(np.dtype('S10').itemsize, 10)
    np.array([['abc', 2], ['long   ', '0123456789']], dtype=np.bytes_)
    assert_equal(np.dtype('S10').itemsize, 10)