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
def test_dm_graph_interface(self):
    m = make_degenerate_solid_phase_model()
    variables = list(m.component_data_objects(pyo.Var))
    constraints = list(m.component_data_objects(pyo.Constraint))
    graph = get_incidence_graph(variables, constraints)
    M, N = (len(constraints), len(variables))
    top_nodes = list(range(M))
    con_dmp, var_dmp = dulmage_mendelsohn(graph, top_nodes=top_nodes)
    con_dmp = tuple(([constraints[i] for i in subset] for subset in con_dmp))
    var_dmp = tuple(([variables[i - M] for i in subset] for subset in var_dmp))
    underconstrained_vars = ComponentSet(m.flow_comp.values())
    underconstrained_vars.add(m.flow)
    underconstrained_cons = ComponentSet(m.flow_eqn.values())
    self.assertEqual(len(var_dmp[0] + var_dmp[1]), len(underconstrained_vars))
    for var in var_dmp[0] + var_dmp[1]:
        self.assertIn(var, underconstrained_vars)
    self.assertEqual(len(con_dmp[2]), len(underconstrained_cons))
    for con in con_dmp[2]:
        self.assertIn(con, underconstrained_cons)
    overconstrained_cons = ComponentSet(m.holdup_eqn.values())
    overconstrained_cons.add(m.density_eqn)
    overconstrained_cons.add(m.sum_eqn)
    overconstrained_vars = ComponentSet(m.x.values())
    overconstrained_vars.add(m.rho)
    self.assertEqual(len(var_dmp[2]), len(overconstrained_vars))
    for var in var_dmp[2]:
        self.assertIn(var, overconstrained_vars)
    self.assertEqual(len(con_dmp[0] + con_dmp[1]), len(overconstrained_cons))
    for con in con_dmp[0] + con_dmp[1]:
        self.assertIn(con, overconstrained_cons)