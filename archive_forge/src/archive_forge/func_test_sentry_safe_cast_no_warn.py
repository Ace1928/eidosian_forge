import warnings
from numba.core import types
from numba.tests.support import TestCase
from numba.typed.typedobjectutils import _sentry_safe_cast
def test_sentry_safe_cast_no_warn(self):
    ok_cases = []
    ok_cases += [(types.int32, types.int64), (types.uint8, types.int32), (types.float32, types.float64), (types.complex64, types.complex128), (types.int32, types.float64), (types.uint8, types.float32), (types.float32, types.complex128), (types.float64, types.complex128), (types.Tuple([types.int32]), types.Tuple([types.int64]))]
    for fromty, toty in ok_cases:
        with self.subTest(fromty=fromty, toty=toty):
            with warnings.catch_warnings(record=True) as w:
                _sentry_safe_cast(fromty, toty)
            self.assertEqual(len(w), 0)