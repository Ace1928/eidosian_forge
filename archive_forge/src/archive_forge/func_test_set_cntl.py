import ctypes
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
def test_set_cntl(self):
    ma57 = MA57Interface()
    ma57.set_cntl(1, 1e-08)
    ma57.set_cntl(2, 1e-12)
    self.assertAlmostEqual(ma57.get_cntl(1), 1e-08)
    self.assertAlmostEqual(ma57.get_cntl(2), 1e-12)