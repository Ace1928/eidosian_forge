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
def test_for_equal_names(self):
    dt = np.dtype([('foo', float), ('bar', float)])
    a = np.zeros(10, dt)
    b = list(a.dtype.names)
    b[0] = 'notfoo'
    a.dtype.names = b
    assert_(a.dtype.names[0] == 'notfoo')
    assert_(a.dtype.names[1] == 'bar')