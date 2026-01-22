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
def test_char_dump(self):
    ca = np.char.array(np.arange(1000, 1010), itemsize=4)
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        with BytesIO() as f:
            pickle.dump(ca, f, protocol=proto)
            f.seek(0)
            ca = np.load(f, allow_pickle=True)