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
def test_diagonal_blocks(self):
    N = 5
    model = make_gas_expansion_model(N)
    igraph = IncidenceGraphInterface()
    variables = []
    variables.extend(model.P.values())
    variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
    constraints = list(model.component_data_objects(pyo.Constraint))
    var_blocks, con_blocks = igraph.get_diagonal_blocks(variables, constraints)
    self.assertEqual(len(var_blocks), N + 1)
    self.assertEqual(len(con_blocks), N + 1)
    for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
        var_set = ComponentSet(vars)
        con_set = ComponentSet(cons)
        if i == 0:
            pred_var_set = ComponentSet([model.P[0]])
            self.assertEqual(pred_var_set, var_set)
            pred_con_set = ComponentSet([model.ideal_gas[0]])
            self.assertEqual(pred_con_set, con_set)
        else:
            pred_var_set = ComponentSet([model.rho[i], model.T[i], model.P[i], model.F[i]])
            pred_con_set = ComponentSet([model.ideal_gas[i], model.expansion[i], model.mbal[i], model.ebal[i]])
            self.assertEqual(pred_var_set, var_set)
            self.assertEqual(pred_con_set, con_set)