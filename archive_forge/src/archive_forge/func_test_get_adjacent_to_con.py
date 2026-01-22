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
def test_get_adjacent_to_con(self):
    m = make_degenerate_solid_phase_model()
    igraph = IncidenceGraphInterface(m)
    adj_vars = igraph.get_adjacent_to(m.density_eqn)
    self.assertEqual(ComponentSet(adj_vars), ComponentSet([m.x[1], m.x[2], m.x[3], m.rho]))