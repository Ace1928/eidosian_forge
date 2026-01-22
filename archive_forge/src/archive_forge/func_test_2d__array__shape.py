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
def test_2d__array__shape(self):

    class T:

        def __array__(self):
            return np.ndarray(shape=(0, 0))

        def __iter__(self):
            return iter([])

        def __getitem__(self, idx):
            raise AssertionError('__getitem__ was called')

        def __len__(self):
            return 0
    t = T()
    arr = np.array([t])
    assert arr.shape == (1, 0, 0)