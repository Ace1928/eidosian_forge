import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_user_code_error_traceback_nopython(self):
    self.check_user_code_error_traceback(flags=no_pyobj_flags)