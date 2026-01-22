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
def test_maps_from_triangularization(self):
    N = 5
    model = make_gas_expansion_model(N)
    igraph = IncidenceGraphInterface()
    variables = []
    variables.extend(model.P.values())
    variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
    constraints = list(model.component_data_objects(pyo.Constraint))
    var_block_map, con_block_map = igraph.map_nodes_to_block_triangular_indices(variables, constraints)
    var_values = set(var_block_map.values())
    con_values = set(con_block_map.values())
    self.assertEqual(len(var_values), N + 1)
    self.assertEqual(len(con_values), N + 1)
    self.assertEqual(var_block_map[model.P[0]], 0)
    for i in model.streams:
        if i != model.streams.first():
            self.assertEqual(var_block_map[model.rho[i]], i)
            self.assertEqual(var_block_map[model.T[i]], i)
            self.assertEqual(var_block_map[model.P[i]], i)
            self.assertEqual(var_block_map[model.F[i]], i)
            self.assertEqual(con_block_map[model.ideal_gas[i]], i)
            self.assertEqual(con_block_map[model.expansion[i]], i)
            self.assertEqual(con_block_map[model.mbal[i]], i)
            self.assertEqual(con_block_map[model.ebal[i]], i)