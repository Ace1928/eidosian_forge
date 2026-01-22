import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def raise_class(exc):

    def raiser(i):
        if i == 1:
            raise exc
        elif i == 2:
            raise ValueError
        elif i == 3:
            raise np.linalg.LinAlgError
        return i
    return raiser