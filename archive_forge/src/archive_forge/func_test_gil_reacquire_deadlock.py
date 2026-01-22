import time
import ctypes
import numpy as np
from numba.tests.support import captured_stdout
from numba import vectorize, guvectorize
import unittest
def test_gil_reacquire_deadlock(self):
    """
        Testing similar issue to #1998 due to GIL reacquiring for Gufunc
        """
    proto = ctypes.CFUNCTYPE(None, ctypes.c_int32)
    characters = 'abcdefghij'

    def bar(x):
        print(characters[x])
    cbar = proto(bar)

    @guvectorize(['(int32, int32[:])'], '()->()', target='parallel', nopython=True)
    def foo(x, out):
        print(x % 10)
        cbar(x % 10)
        out[0] = x * 2
    for nelem in [1, 10, 100, 1000]:
        a = np.arange(nelem, dtype=np.int32)
        acopy = a.copy()
        with captured_stdout() as buf:
            got = foo(a)
        stdout = buf.getvalue()
        buf.close()
        got_output = sorted(map(lambda x: x.strip(), stdout.splitlines()))
        expected_output = [str(x % 10) for x in range(nelem)]
        expected_output += [characters[x % 10] for x in range(nelem)]
        expected_output = sorted(expected_output)
        self.assertEqual(got_output, expected_output)
        np.testing.assert_equal(got, 2 * acopy)