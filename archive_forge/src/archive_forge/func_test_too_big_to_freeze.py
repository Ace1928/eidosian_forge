import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_too_big_to_freeze(self):
    """
        Test issue https://github.com/numba/numba/issues/2188 where freezing
        a constant array into the code that's prohibitively long and consumes
        too much RAM.
        """

    def test(biggie):
        expect = np.copy(biggie)
        self.assertEqual(typeof(biggie), typeof(expect))

        def pyfunc():
            return biggie
        cfunc = njit(())(pyfunc)
        self.assertLess(len(cfunc.inspect_llvm(())), biggie.nbytes)
        out = cfunc()
        self.assertIs(biggie, out)
        del out
        biggie = None
        out = cfunc()
        np.testing.assert_equal(expect, out)
        self.assertEqual(typeof(expect), typeof(out))
    nelem = 10 ** 7
    c_array = np.arange(nelem).reshape(nelem)
    f_array = np.asfortranarray(np.random.random((2, nelem // 2)))
    self.assertEqual(typeof(c_array).layout, 'C')
    self.assertEqual(typeof(f_array).layout, 'F')
    test(c_array)
    test(f_array)