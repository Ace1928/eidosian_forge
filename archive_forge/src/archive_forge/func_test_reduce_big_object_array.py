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
def test_reduce_big_object_array(self):
    oldsize = np.setbufsize(10 * 16)
    a = np.array([None] * 161, object)
    assert_(not np.any(a))
    np.setbufsize(oldsize)