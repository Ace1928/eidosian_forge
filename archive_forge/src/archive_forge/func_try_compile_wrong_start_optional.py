from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
def try_compile_wrong_start_optional(*args):
    wrong_sig_optional = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.float64), types.Optional(types.intp))
    njit([wrong_sig_optional])(rfind_with_start_end_usecase)