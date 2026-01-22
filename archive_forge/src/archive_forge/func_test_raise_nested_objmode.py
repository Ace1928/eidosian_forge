import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_raise_nested_objmode(self):
    self.check_raise_nested(force_pyobj_flags, forceobj=True)