import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def raise_instance_runtime_args(exc):

    def raiser(i, arg):
        if i == 1:
            raise exc(arg, 1)
        elif i == 2:
            raise ValueError(arg, 2)
        elif i == 3:
            raise np.linalg.LinAlgError(arg, 3)
        return i
    return raiser