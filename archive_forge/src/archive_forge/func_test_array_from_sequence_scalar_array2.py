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
def test_array_from_sequence_scalar_array2(self):
    t = np.array([np.array([]), np.array(0, object)], dtype=object)
    assert_equal(t.shape, (2,))
    assert_equal(t.dtype, np.dtype(object))