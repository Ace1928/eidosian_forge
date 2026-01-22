import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_zero_coef(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    expr = 0 * m.x[1] + 0 * m.x[1] * m.x[2] + 0 * pyo.exp(m.x[3])
    variables = self._get_incident_variables(expr)
    self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))