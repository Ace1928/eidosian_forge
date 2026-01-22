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
def test_void_item_memview(self):
    va = np.zeros(10, 'V4')
    x = va[:1].item()
    va[0] = b'\xff\xff\xff\xff'
    del va
    assert_equal(x, b'\x00\x00\x00\x00')