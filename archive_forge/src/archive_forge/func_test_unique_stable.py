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
def test_unique_stable(self):
    v = np.array(([0] * 5 + [1] * 6 + [2] * 6) * 4)
    res = np.unique(v, return_index=True)
    tgt = (np.array([0, 1, 2]), np.array([0, 5, 11]))
    assert_equal(res, tgt)