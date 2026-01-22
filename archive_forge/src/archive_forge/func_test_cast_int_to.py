from textwrap import dedent
from numba import njit
from numba import int32
from numba.extending import register_jitable
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.typed import listobject, List
def test_cast_int_to(self):
    self.check_good(types.int32, types.float32)
    self.check_good(types.int32, types.float64)
    self.check_good(types.int32, types.complex128)
    self.check_good(types.int64, types.complex128)
    self.check_bad(types.int32, types.complex64)
    self.check_good(types.int8, types.complex64)