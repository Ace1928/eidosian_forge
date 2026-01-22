from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_getitem_multiple_slice_start_out_of_range(self):

    @njit
    def foo():
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        n = l[10:]
        return len(n)
    self.assertEqual(foo(), 0)