import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_parallel_matvec(self):
    err = parallel_matvec.main()
    self.assertLessEqual(err, 1e-15)