import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_assert_statement_nopython(self):
    self.check_assert_statement(flags=no_pyobj_flags)