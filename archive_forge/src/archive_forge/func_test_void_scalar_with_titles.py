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
def test_void_scalar_with_titles(self):
    data = [('john', 4), ('mary', 5)]
    dtype1 = [(('source:yy', 'name'), 'O'), (('source:xx', 'id'), int)]
    arr = np.array(data, dtype=dtype1)
    assert_(arr[0][0] == 'john')
    assert_(arr[0][1] == 4)