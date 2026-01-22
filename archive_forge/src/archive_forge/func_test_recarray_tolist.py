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
def test_recarray_tolist(self):
    buf = np.zeros(40, dtype=np.int8)
    a = np.recarray(2, formats='i4,f8,f8', names='id,x,y', buf=buf)
    b = a.tolist()
    assert_(a[0].tolist() == b[0])
    assert_(a[1].tolist() == b[1])