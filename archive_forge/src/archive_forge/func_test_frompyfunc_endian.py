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
def test_frompyfunc_endian(self):
    from math import radians
    uradians = np.frompyfunc(radians, 1, 1)
    big_endian = np.array([83.4, 83.5], dtype='>f8')
    little_endian = np.array([83.4, 83.5], dtype='<f8')
    assert_almost_equal(uradians(big_endian).astype(float), uradians(little_endian).astype(float))