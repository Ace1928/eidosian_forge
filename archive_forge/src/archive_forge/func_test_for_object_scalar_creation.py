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
def test_for_object_scalar_creation(self):
    a = np.object_()
    b = np.object_(3)
    b2 = np.object_(3.0)
    c = np.object_([4, 5])
    d = np.object_([None, {}, []])
    assert_(a is None)
    assert_(type(b) is int)
    assert_(type(b2) is float)
    assert_(type(c) is np.ndarray)
    assert_(c.dtype == object)
    assert_(d.dtype == object)