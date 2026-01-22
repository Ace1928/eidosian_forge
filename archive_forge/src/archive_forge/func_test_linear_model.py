import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
def test_linear_model(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var([1, 2, 3])
    m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
    m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)
    repn = LinearStandardFormCompiler().write(m)
    self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
    self.assertTrue(np.all(repn.A == np.array([[-1, -2, 0], [0, 1, 4]])))
    self.assertTrue(np.all(repn.rhs == np.array([-3, 5])))