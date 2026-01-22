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
def test_string_sort_with_zeros(self):
    x = np.frombuffer(b'\x00\x02\x00\x01', dtype='|S2')
    y = np.frombuffer(b'\x00\x01\x00\x02', dtype='|S2')
    assert_array_equal(np.sort(x, kind='q'), y)