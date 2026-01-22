import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_tocsc(self):
    with self.assertRaises(Exception) as context:
        self.square_mpi_mat.tocsc()