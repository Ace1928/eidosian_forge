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
def test_triangularize_submatrix(self):
    N = 5
    model = make_gas_expansion_model(N)
    igraph = IncidenceGraphInterface(model)
    variables = []
    half = N // 2
    variables.extend((model.P[i] for i in model.streams if i >= half))
    variables.extend((model.T[i] for i in model.streams if i > half))
    variables.extend((model.rho[i] for i in model.streams if i > half))
    variables.extend((model.F[i] for i in model.streams if i > half))
    constraints = []
    constraints.extend((model.ideal_gas[i] for i in model.streams if i >= half))
    constraints.extend((model.expansion[i] for i in model.streams if i > half))
    constraints.extend((model.mbal[i] for i in model.streams if i > half))
    constraints.extend((model.ebal[i] for i in model.streams if i > half))
    var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
    partition = [list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)]
    self.assertEqual(len(partition), N - half + 1)
    for i in model.streams:
        idx = i - half
        variables = ComponentSet([var for var, _ in partition[idx]])
        constraints = ComponentSet([con for _, con in partition[idx]])
        if i == half:
            self.assertEqual(variables, ComponentSet([model.P[half]]))
        elif i > half:
            pred_var = ComponentSet([model.rho[i], model.T[i], model.P[i], model.F[i]])
            pred_con = ComponentSet([model.ideal_gas[i], model.expansion[i], model.mbal[i], model.ebal[i]])
            self.assertEqual(variables, pred_var)
            self.assertEqual(constraints, pred_con)