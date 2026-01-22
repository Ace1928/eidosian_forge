from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_setitem_singleton_negative_index(self):

    @njit
    def foo(n):
        l = listobject.new_list(int32)
        l.append(0)
        l[0] = n
        return l[-1]
    for i in (0, 1, 2, 100):
        self.assertEqual(foo(i), i)