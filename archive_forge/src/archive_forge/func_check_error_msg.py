import random
import numpy as np
from numba.tests.support import TestCase, captured_stdout
from numba import njit, literally
from numba.core import types
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.np.unsafe.ndarray import to_fixed_tuple, empty_inferred
from numba.core.unsafe.bytes import memcpy_region
from numba.core.unsafe.refcount import dump_refcount
from numba.cpython.unsafe.numbers import trailing_zeros, leading_zeros
from numba.core.errors import TypingError
def check_error_msg(self, func):
    cfunc = njit(lambda *x: func(*x))
    func_name = func._name
    unsupported_types = filter(lambda x: not isinstance(x, types.Integer), types.number_domain)
    for typ in sorted(unsupported_types, key=str):
        with self.assertRaises(TypingError) as e:
            cfunc(typ(2))
        self.assertIn("{} is only defined for integers, but value passed was '{}'.".format(func_name, typ), str(e.exception))

    def check(args, string):
        with self.assertRaises((TypingError, TypeError)) as e:
            cfunc(*args)
        self.assertIn('{}() '.format(func_name), str(e.exception))
    check((1, 2), 'takes 2 positional arguments but 3 were given')
    check((), 'missing 1 required positional argument')