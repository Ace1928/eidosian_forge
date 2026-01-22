import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def make_test_model(self):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[1, 2, 3, 4])
    m.v = pyo.Var(m.I, bounds=(0, None))
    m.eq1 = pyo.Constraint(expr=m.v[1] ** 2 + m.v[2] ** 2 == 1.0)
    m.eq2 = pyo.Constraint(expr=m.v[1] + 2.0 == m.v[3])
    m.ineq1 = pyo.Constraint(expr=m.v[2] - m.v[3] ** 0.5 + m.v[4] ** 2 <= 1.0)
    m.ineq2 = pyo.Constraint(expr=m.v[2] * m.v[4] >= 1.0)
    m.ineq3 = pyo.Constraint(expr=m.v[1] >= m.v[4] ** 4)
    m.obj = pyo.Objective(expr=-m.v[1] - m.v[2] + m.v[3] ** 2 + m.v[4] ** 2)
    return m