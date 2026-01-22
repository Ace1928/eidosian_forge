from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_index_typing_error_start(self):
    self.disable_leak_check()

    @njit
    def foo():
        l = listobject.new_list(int32)
        l.append(0)
        return l.index(0, start='a')
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('start argument for index must be an integer', str(raises.exception))