import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_return_class_is_ndarray_numpy(self):
    pyfunc = np_allclose
    cfunc = jit(nopython=True)(pyfunc)

    class Foo(np.ndarray):

        def __new__(cls, *args, **kwargs):
            return np.array(*args, **kwargs).view(cls)
    a = Foo([1])
    self.assertTrue(type(cfunc(a, a)) is bool)