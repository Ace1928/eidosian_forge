from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_make_immutable_is_immutable(self):

    @njit
    def foo():
        l = make_test_list()
        l._make_immutable()
        return l._is_mutable()
    self.assertFalse(foo())