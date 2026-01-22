from numba import jit
import unittest
import numpy as np
import copy
from numba.tests.support import MemoryLeakMixin
def test_chain3(self):
    from numba.tests.chained_assign_usecases import chain3
    args = [[np.array([0]), np.array([1.5])], [np.array([0.5]), np.array([1])]]
    self._test_template(chain3, args)