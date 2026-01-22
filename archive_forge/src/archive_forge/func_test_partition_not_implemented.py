import itertools
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
def test_partition_not_implemented(self):
    fcn = ImplicitFunction1()
    variables = fcn.get_variables()
    parameters = fcn.get_parameters()
    equations = fcn.get_equations()
    msg = 'has not implemented'
    with self.assertRaisesRegex(NotImplementedError, msg):
        solver = DecomposedImplicitFunctionBase(variables, equations, parameters)