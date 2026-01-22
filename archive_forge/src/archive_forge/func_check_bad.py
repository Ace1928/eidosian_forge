from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def check_bad(self, fromty, toty):
    with self.assertRaises(TypingError) as raises:
        TestItemCasting.foo(fromty, toty)
    self.assertIn('cannot safely cast {fromty} to {toty}'.format(**locals()), str(raises.exception))