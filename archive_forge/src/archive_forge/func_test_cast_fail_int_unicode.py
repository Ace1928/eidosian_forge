from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_cast_fail_int_unicode(self):

    @njit
    def foo():
        l = listobject.new_list(types.unicode_type)
        l.append(int32(0))
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('Cannot cast int32 to unicode_type', str(raises.exception))