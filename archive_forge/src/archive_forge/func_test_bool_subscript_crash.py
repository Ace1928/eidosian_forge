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
def test_bool_subscript_crash(self):
    c = np.rec.array([(1, 2, 3), (4, 5, 6)])
    masked = c[np.array([True, False])]
    base = masked.base
    del masked, c
    base.dtype