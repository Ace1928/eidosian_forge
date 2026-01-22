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
def test_dot_alignment_sse2(self):
    x = np.zeros((30, 40))
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        y = pickle.loads(pickle.dumps(x, protocol=proto))
        z = np.ones((1, y.shape[0]))
        np.dot(z, y)