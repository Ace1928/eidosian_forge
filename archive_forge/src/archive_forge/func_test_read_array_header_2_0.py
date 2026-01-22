import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_read_array_header_2_0():
    s = BytesIO()
    arr = np.ones((3, 6), dtype=float)
    format.write_array(s, arr, version=(2, 0))
    s.seek(format.MAGIC_LEN)
    shape, fortran, dtype = format.read_array_header_2_0(s)
    assert_(s.tell() % format.ARRAY_ALIGN == 0)
    assert_((shape, fortran, dtype) == ((3, 6), False, float))