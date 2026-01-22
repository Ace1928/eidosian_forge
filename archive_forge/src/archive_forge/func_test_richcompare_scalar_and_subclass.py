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
def test_richcompare_scalar_and_subclass(self):

    class Foo(np.ndarray):

        def __eq__(self, other):
            return 'OK'
    x = np.array([1, 2, 3]).view(Foo)
    assert_equal(10 == x, 'OK')
    assert_equal(np.int32(10) == x, 'OK')
    assert_equal(np.array([10]) == x, 'OK')