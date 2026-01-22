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
def test_issue5565_literal_getitem(self):
    Dummy, DummyType = (self.Dummy, self.DummyType)
    MAGIC_NUMBER = 12321

    @overload(operator.getitem)
    def dummy_getitem_ovld(self, idx):
        if not isinstance(self, DummyType):
            return None
        if isinstance(idx, types.StringLiteral):

            def dummy_getitem_impl(self, idx):
                return MAGIC_NUMBER
            return dummy_getitem_impl
        if isinstance(idx, types.UnicodeType):

            def dummy_getitem_impl(self, idx):
                return literally(idx)
            return dummy_getitem_impl
        return None

    @njit
    def test_impl(x, y):
        return Dummy(x)[y]
    var = 'abc'
    self.assertEqual(test_impl(1, var), MAGIC_NUMBER)