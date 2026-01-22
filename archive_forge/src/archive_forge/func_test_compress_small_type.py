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
def test_compress_small_type(self):
    import numpy as np
    a = np.array([[1, 2], [3, 4]])
    b = np.zeros((2, 1), dtype=np.single)
    try:
        a.compress([True, False], axis=1, out=b)
        raise AssertionError('compress with an out which cannot be safely casted should not return successfully')
    except TypeError:
        pass