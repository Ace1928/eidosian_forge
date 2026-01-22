from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_getitem_singleton_negtive_index(self):

    @njit
    def foo(n):
        l = listobject.new_list(int32)
        l.append(n)
        return l[-1]
    self.assertEqual(foo(0), 0)