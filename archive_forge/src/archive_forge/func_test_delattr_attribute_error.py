import unittest
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
def test_delattr_attribute_error(self):
    pyfunc = delattr_usecase
    cfunc = jit((types.pyobject,), forceobj=True)(pyfunc)
    with self.assertRaises(AttributeError):
        cfunc(C())