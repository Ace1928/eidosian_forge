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
def test_mem_fromiter_invalid_dtype_string(self):
    x = [1, 2, 3]
    assert_raises(ValueError, np.fromiter, [xi for xi in x], dtype='S')