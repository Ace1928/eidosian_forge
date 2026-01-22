import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
@unittest.skipUnless(cyipopt_available, 'CyIpopt is not available')
def test_implicit_function_1_with_cyipopt(self):
    self._test_implicit_function_1(solver_class=CyIpoptSolverWrapper)