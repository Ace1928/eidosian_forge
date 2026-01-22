from pyomo.contrib.pynumero.dependencies import numpy_available, scipy_available
import pyomo.common.unittest as unittest
import numpy as np
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.linalg.mumps_interface import mumps_available
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU
import pyomo.environ as pe
from pyomo.contrib.pynumero.examples import (
@unittest.skipIf(not mumps_available, 'mumps example needs pymumps')
def test_mumps_example(self):
    mumps_example.main()