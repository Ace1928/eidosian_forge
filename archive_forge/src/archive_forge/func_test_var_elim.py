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
def test_var_elim(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3, 4])
    m.eq1 = pyo.Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2 == 1)
    m.eq2 = pyo.Constraint(expr=pyo.sqrt(m.x[1]) + pyo.exp(m.x[3]) == 1)
    m.eq3 = pyo.Constraint(expr=m.x[3] + m.x[2] + m.x[4] == 1)
    m.eq4 = pyo.Constraint(expr=m.x[1] == 5 * m.x[2])
    igraph = IncidenceGraphInterface(m)
    for adj_con in igraph.get_adjacent_to(m.x[1]):
        for adj_var in igraph.get_adjacent_to(m.eq4):
            igraph.add_edge(adj_var, adj_con)
    igraph.remove_nodes([m.x[1], m.eq4])
    assert ComponentSet(igraph.variables) == ComponentSet([m.x[2], m.x[3], m.x[4]])
    assert ComponentSet(igraph.constraints) == ComponentSet([m.eq1, m.eq2, m.eq3])
    self.assertEqual(7, igraph.n_edges)
    assert m.x[2] in ComponentSet(igraph.get_adjacent_to(m.eq1))
    assert m.x[2] in ComponentSet(igraph.get_adjacent_to(m.eq2))