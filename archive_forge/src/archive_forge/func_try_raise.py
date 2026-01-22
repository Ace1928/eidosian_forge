import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
@njit
def try_raise(a):
    try:
        raise_(a)
    except Exception:
        pass
    return a + 1