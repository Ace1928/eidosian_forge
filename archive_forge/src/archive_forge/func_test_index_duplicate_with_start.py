from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_index_duplicate_with_start(self):

    @njit
    def foo(start):
        l = listobject.new_list(int32)
        for _ in range(10, 20):
            l.append(1)
        return l.index(1, start)
    for i in range(10):
        self.assertEqual(foo(i), i)