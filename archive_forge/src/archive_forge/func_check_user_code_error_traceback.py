import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_user_code_error_traceback(self, flags):
    pyfunc = ude_bug_usecase
    cfunc = jit((), **flags)(pyfunc)
    self.check_against_python(flags, pyfunc, cfunc, TypeError)