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
def test_triangularize(self):
    N = 5
    model = make_gas_expansion_model(N)
    igraph = IncidenceGraphInterface()
    variables = []
    variables.extend(model.P.values())
    variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
    constraints = list(model.component_data_objects(pyo.Constraint))
    var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
    partition = [list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)]
    self.assertEqual(len(partition), N + 1)
    for i in model.streams:
        variables = ComponentSet([var for var, _ in partition[i]])
        constraints = ComponentSet([con for _, con in partition[i]])
        if i == model.streams.first():
            self.assertEqual(variables, ComponentSet([model.P[0]]))
        else:
            pred_vars = ComponentSet([model.rho[i], model.T[i], model.P[i], model.F[i]])
            pred_cons = ComponentSet([model.ideal_gas[i], model.expansion[i], model.mbal[i], model.ebal[i]])
            self.assertEqual(pred_vars, variables)
            self.assertEqual(pred_cons, constraints)