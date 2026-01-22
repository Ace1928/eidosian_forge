import operator
from numba import njit, literally
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError
from numba.core.extending import lower_builtin
from numba.core.extending import models, register_model
from numba.core.extending import make_attribute_wrapper
from numba.core.extending import type_callable
from numba.core.extending import overload
from numba.core.extending import typeof_impl
import unittest
def test_unboxing(self):
    """A test for the unboxing logic on unknown type
        """
    Dummy = self.Dummy

    @njit
    def foo(x):
        bar(Dummy(x))

    @njit(no_cpython_wrapper=False)
    def bar(dummy_obj):
        pass
    foo(123)
    with self.assertRaises(TypeError) as raises:
        bar(Dummy(123))
    self.assertIn("can't unbox Dummy type", str(raises.exception))