from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_index_multiple_value_error(self):
    self.disable_leak_check()

    @njit
    def foo():
        l = listobject.new_list(int32)
        for j in range(10, 20):
            l.append(j)
        return l.index(23)
    with self.assertRaises(ValueError) as raises:
        foo()
    self.assertIn('item not in list', str(raises.exception))