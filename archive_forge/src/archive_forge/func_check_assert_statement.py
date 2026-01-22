import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_assert_statement(self, flags):
    pyfunc = assert_usecase
    cfunc = jit((types.int32,), **flags)(pyfunc)
    cfunc(1)
    self.check_against_python(flags, pyfunc, cfunc, AssertionError, 2)