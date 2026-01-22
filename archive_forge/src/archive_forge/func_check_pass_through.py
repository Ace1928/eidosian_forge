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
def check_pass_through(jitted, expect_same, params):
    returned = jitted(**params)
    if expect_same:
        self.assertTrue(returned is params['a'])
    else:
        self.assertTrue(returned is not params['a'])
        np.testing.assert_allclose(returned, params['a'])
        self.assertTrue(returned.dtype == params['dtype'])