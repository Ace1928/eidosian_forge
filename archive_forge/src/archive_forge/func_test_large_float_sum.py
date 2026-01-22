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
def test_large_float_sum(self):
    a = np.arange(10000, dtype='f')
    assert_equal(a.sum(dtype='d'), a.astype('d').sum())