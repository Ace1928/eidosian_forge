import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_variable_minus_itself(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    expr = m.x[1] + m.x[2] * m.x[3] - m.x[1]
    variables = self._get_incident_variables(expr)
    var_set = ComponentSet(variables)
    self.assertEqual(var_set, ComponentSet(m.x[:]))