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
def test__array_interface__descr(self):
    dt = np.dtype(dict(names=['a', 'b'], offsets=[0, 0], formats=[np.int64, np.int64]))
    descr = np.array((1, 1), dtype=dt).__array_interface__['descr']
    assert descr == [('', '|V8')]