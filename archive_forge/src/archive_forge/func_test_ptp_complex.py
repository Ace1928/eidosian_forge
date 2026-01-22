from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_ptp_complex(self):
    pyfunc = array_ptp_global
    cfunc = jit(nopython=True)(pyfunc)

    def check(a):
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)

    def make_array(real_nan=False, imag_nan=False):
        real = np.linspace(-4, 4, 25)
        if real_nan:
            real[4:9] = np.nan
        imag = np.linspace(-5, 5, 25)
        if imag_nan:
            imag[7:12] = np.nan
        return (real + 1j * imag).reshape(5, 5)
    for real_nan, imag_nan in product([True, False], repeat=2):
        comp = make_array(real_nan, imag_nan)
        check(comp)
    real = np.ones(8)
    imag = np.arange(-4, 4)
    comp = real + 1j * imag
    check(comp)
    comp = real - 1j * imag
    check(comp)
    comp = np.full((4, 4), fill_value=1 - 1j)
    check(comp)