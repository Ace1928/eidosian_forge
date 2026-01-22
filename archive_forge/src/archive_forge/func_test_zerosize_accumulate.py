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
def test_zerosize_accumulate(self):
    """Ticket #1733"""
    x = np.array([[42, 0]], dtype=np.uint32)
    assert_equal(np.add.accumulate(x[:-1, 0]), [])