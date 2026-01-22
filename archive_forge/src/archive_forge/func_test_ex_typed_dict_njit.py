import unittest
from numba.tests.support import captured_stdout
def test_ex_typed_dict_njit(self):
    with captured_stdout():
        import numpy as np
        from numba import njit
        from numba.core import types
        from numba.typed import Dict
        float_array = types.float64[:]

        @njit
        def foo():
            d = Dict.empty(key_type=types.unicode_type, value_type=float_array)
            d['posx'] = np.arange(3).astype(np.float64)
            d['posy'] = np.arange(3, 6).astype(np.float64)
            return d
        d = foo()
        print(d)
    np.testing.assert_array_equal(d['posx'], [0, 1, 2])
    np.testing.assert_array_equal(d['posy'], [3, 4, 5])