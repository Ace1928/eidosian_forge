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
@needs_blas
def test_corrcoef_explicit_arguments(self):
    pyfunc = corrcoef
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-14)
    x = self.rnd.randn(105).reshape(15, 7)
    y_choices = (None, x[::-1])
    rowvar_choices = (False, True)
    for y, rowvar in itertools.product(y_choices, rowvar_choices):
        params = {'x': x, 'y': y, 'rowvar': rowvar}
        _check(params)