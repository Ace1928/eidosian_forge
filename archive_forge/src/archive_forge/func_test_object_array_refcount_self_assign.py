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
def test_object_array_refcount_self_assign(self):

    class VictimObject:
        deleted = False

        def __del__(self):
            self.deleted = True
    d = VictimObject()
    arr = np.zeros(5, dtype=np.object_)
    arr[:] = d
    del d
    arr[:] = arr
    assert_(not arr[0].deleted)
    arr[:] = arr
    assert_(not arr[0].deleted)