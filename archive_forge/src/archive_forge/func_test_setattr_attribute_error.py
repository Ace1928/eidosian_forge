import unittest
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
def test_setattr_attribute_error(self):
    pyfunc = setattr_usecase
    cfunc = jit((types.pyobject, types.int32), forceobj=True)(pyfunc)
    with self.assertRaises(AttributeError):
        cfunc(object(), 123)