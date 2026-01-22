from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_list_insert_typing_error(self):
    self.disable_leak_check()

    @njit
    def foo():
        l = listobject.new_list(int32)
        l.insert('a', 0)
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('list insert indices must be integers', str(raises.exception))