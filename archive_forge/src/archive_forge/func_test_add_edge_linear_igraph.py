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
def test_add_edge_linear_igraph(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3, 4])
    m.eq1 = pyo.Constraint(expr=m.x[1] + m.x[3] == 1)
    m.eq2 = pyo.Constraint(expr=m.x[2] + pyo.sqrt(m.x[1]) + pyo.exp(m.x[3]) == 1)
    m.eq3 = pyo.Constraint(expr=m.x[4] ** 2 + m.x[1] ** 3 + m.x[2] ** 2 == 1)
    igraph = IncidenceGraphInterface(m, linear_only=True)
    msg = 'is not a variable in the incidence graph'
    with self.assertRaisesRegex(RuntimeError, msg):
        igraph.add_edge(m.x[4], m.eq2)