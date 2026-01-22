import gc
import weakref
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
import unittest
def test_local_func_lifetime(self):
    self.check_local_func_lifetime(forceobj=True)