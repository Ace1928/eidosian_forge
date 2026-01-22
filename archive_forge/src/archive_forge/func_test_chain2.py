from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_chain2(self):
    from numba.tests.chained_assign_usecases import chain2
    args = [[3], [3.0]]
    self._test_template(chain2, args)