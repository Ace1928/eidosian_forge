import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.pyomo_ext_cyipopt import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
def evaluate_derivatives(self):
    jac = [[1, -self._F ** 2, 0, -2 * self._c1 * self._F], [1, -self._F ** 2, -self._F ** 2, -2 * self._F * (self._c1 + self._c2)]]
    jac = np.asarray(jac, dtype=np.float64)
    return spa.coo_matrix(jac)