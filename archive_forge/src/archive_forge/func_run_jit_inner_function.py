import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def run_jit_inner_function(self, **jitargs):

    def mult_10(a):
        return a * 10
    c_mult_10 = jit('intp(intp)', **jitargs)(mult_10)
    c_mult_10.disable_compile()

    def do_math(x):
        return c_mult_10(x + 4)
    c_do_math = jit('intp(intp)', **jitargs)(do_math)
    c_do_math.disable_compile()
    with self.assertRefCount(c_do_math, c_mult_10):
        self.assertEqual(c_do_math(1), 50)