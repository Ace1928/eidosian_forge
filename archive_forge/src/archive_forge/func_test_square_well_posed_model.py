import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import get_structural_incidence_matrix
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_square_well_posed_model(self):
    N = 4
    m = make_gas_expansion_model(N)
    m.F[0].fix()
    m.rho[0].fix()
    m.T[0].fix()
    variables = [v for v in m.component_data_objects(pyo.Var) if not v.fixed]
    constraints = list(m.component_data_objects(pyo.Constraint))
    imat = get_structural_incidence_matrix(variables, constraints)
    N, M = imat.shape
    self.assertEqual(N, M)
    row_partition, col_partition = dulmage_mendelsohn(imat)
    self.assertEqual(len(row_partition[0]), 0)
    self.assertEqual(len(row_partition[1]), 0)
    self.assertEqual(len(row_partition[2]), 0)
    self.assertEqual(len(col_partition[0]), 0)
    self.assertEqual(len(col_partition[1]), 0)
    self.assertEqual(len(col_partition[2]), 0)
    self.assertEqual(len(row_partition[3]), M)
    self.assertEqual(len(col_partition[3]), N)