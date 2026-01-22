import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_gen8(self, **jit_args):
    pyfunc = gen8
    cfunc = jit(**jit_args)(pyfunc)

    def check(*args, **kwargs):
        self.check_generator(pyfunc(*args, **kwargs), cfunc(*args, **kwargs))
    check(2, 3)
    check(4)
    check(y=5)
    check(x=6, b=True)