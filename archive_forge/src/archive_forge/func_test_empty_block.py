import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_empty_block(self):
    m = self.square_mpi_mat
    self.assertFalse(m.is_empty_block(0, 0, False))
    self.assertFalse(m.is_empty_block(1, 1, False))
    self.assertTrue(m.is_empty_block(0, 1, False))
    self.assertTrue(m.is_empty_block(1, 0, False))