from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_count_mutiple(self):

    @njit
    def foo(i):
        l = listobject.new_list(int32)
        for j in [11, 12, 12, 13, 13, 13]:
            l.append(j)
        return l.count(i)
    self.assertEqual(foo(10), 0)
    self.assertEqual(foo(11), 1)
    self.assertEqual(foo(12), 2)
    self.assertEqual(foo(13), 3)