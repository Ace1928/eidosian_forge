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
def test_format_on_flex_array_element(self):
    dt = np.dtype([('date', '<M8[D]'), ('val', '<f8')])
    arr = np.array([('2000-01-01', 1)], dt)
    formatted = '{0}'.format(arr[0])
    assert_equal(formatted, str(arr[0]))