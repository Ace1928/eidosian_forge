import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_fixed_zero_coefficient_linear_only(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    expr = m.x[1] * m.x[2] + 2 * m.x[3]
    m.x[2].fix(0)
    variables = get_incident_variables(expr, method=IncidenceMethod.standard_repn, linear_only=True)
    self.assertEqual(len(variables), 1)
    self.assertIs(variables[0], m.x[3])