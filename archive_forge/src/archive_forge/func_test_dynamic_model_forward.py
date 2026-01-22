import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_dynamic_model_forward(self):
    m = make_dynamic_model(nfe=5, scheme='FORWARD')
    m.height[0].fix()
    constraints = list(m.component_data_objects(pyo.Constraint, active=True))
    variables = list(_generate_variables_in_constraints(constraints))
    con_coord_map = ComponentMap(((con, i) for i, con in enumerate(constraints)))
    var_coord_map = ComponentMap(((var, i) for i, var in enumerate(variables)))
    coo = get_structural_incidence_matrix(variables, constraints)
    row_blocks, col_blocks = get_independent_submatrices(coo)
    rc_blocks = [(tuple(rows), tuple(cols)) for rows, cols in zip(row_blocks, col_blocks)]
    self.assertEqual(len(rc_blocks), 1)