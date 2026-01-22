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
def test_noncontiguous_fill(self):
    a = np.zeros((5, 3))
    b = a[:, :2]

    def rs():
        b.shape = (10,)
    assert_raises(AttributeError, rs)