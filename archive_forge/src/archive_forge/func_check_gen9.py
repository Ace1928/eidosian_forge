import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def check_gen9(self, **kwargs):
    pyfunc = gen_bool
    cr = jit((), **kwargs)(pyfunc)
    pygen = pyfunc()
    cgen = cr()
    self.check_generator(pygen, cgen)