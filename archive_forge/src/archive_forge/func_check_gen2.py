import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_gen2(self, **kwargs):
    pyfunc = gen2
    cr = jit((types.int32,), **kwargs)(pyfunc)
    pygen = pyfunc(8)
    cgen = cr(8)
    self.check_generator(pygen, cgen)