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
def test_chararray_rstrip(self):
    x = np.chararray((1,), 5)
    x[0] = b'a   '
    x = x.rstrip()
    assert_equal(x[0], b'a')