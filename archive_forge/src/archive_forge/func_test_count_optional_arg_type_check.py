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
def test_count_optional_arg_type_check(self):
    pyfunc = count_with_start_end_usecase

    def try_compile_bad_optional(*args):
        bad_sig = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.float64), types.Optional(types.float64))
        njit([bad_sig])(pyfunc)
    with self.assertRaises(TypingError) as raises:
        try_compile_bad_optional('tú quis?', 'tú', 1.1, 1.1)
    self.assertIn('The slice indices must be an Integer or None', str(raises.exception))
    error_msg = '%s\n%s' % ("'{0}'.py_count('{1}', {2}, {3}) = {4}", "'{0}'.c_count_op('{1}', {2}, {3}) = {5}")
    sig_optional = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.int64), types.Optional(types.int64))
    cfunc_optional = njit([sig_optional])(pyfunc)
    py_result = pyfunc('tú quis?', 'tú', 0, 8)
    c_result = cfunc_optional('tú quis?', 'tú', 0, 8)
    self.assertEqual(py_result, c_result, error_msg.format('tú quis?', 'tú', 0, 8, py_result, c_result))