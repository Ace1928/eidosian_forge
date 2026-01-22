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
def test_take_output(self):
    x = np.arange(12).reshape((3, 4))
    a = np.take(x, [0, 2], axis=1)
    b = np.zeros_like(a)
    np.take(x, [0, 2], axis=1, out=b)
    assert_array_equal(a, b)