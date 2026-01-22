import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_changing_number_type(self):
    pyfunc = gen_changing_number_type
    expected = list(pyfunc())
    got = list(njit(pyfunc)())
    self.assertEqual(expected, got)