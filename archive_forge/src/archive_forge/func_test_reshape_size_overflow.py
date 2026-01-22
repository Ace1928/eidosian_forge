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
def test_reshape_size_overflow(self):
    a = np.ones(20)[::2]
    if np.dtype(np.intp).itemsize == 8:
        new_shape = (2, 13, 419, 691, 823, 2977518503)
    else:
        new_shape = (2, 7, 7, 43826197)
    assert_raises(ValueError, a.reshape, new_shape)