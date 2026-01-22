import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_read_magic():
    s1 = BytesIO()
    s2 = BytesIO()
    arr = np.ones((3, 6), dtype=float)
    format.write_array(s1, arr, version=(1, 0))
    format.write_array(s2, arr, version=(2, 0))
    s1.seek(0)
    s2.seek(0)
    version1 = format.read_magic(s1)
    version2 = format.read_magic(s2)
    assert_(version1 == (1, 0))
    assert_(version2 == (2, 0))
    assert_(s1.tell() == format.MAGIC_LEN)
    assert_(s2.tell() == format.MAGIC_LEN)