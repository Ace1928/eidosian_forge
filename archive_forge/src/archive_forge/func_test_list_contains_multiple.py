from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_contains_multiple(self):

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        return i in l
    for i in range(10, 20):
        self.assertTrue(foo(i))
    for i in range(20, 30):
        self.assertFalse(foo(i))