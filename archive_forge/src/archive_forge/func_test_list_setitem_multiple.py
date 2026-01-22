from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_setitem_multiple(self):

    @njit
    def foo(i, n):
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        l[i] = n
        return l[i]
    for i, n in zip(range(0, 10), range(20, 30)):
        self.assertEqual(foo(i, n), n)