import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_incidence_with_fixed_variable(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=1.0)
    expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
    m.x[2].fix()
    variables = self._get_incident_variables(expr)
    var_set = ComponentSet(variables)
    self.assertEqual(var_set, ComponentSet([m.x[1], m.x[3]]))